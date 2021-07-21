import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.rpc as rpc
import torch.optim as optim
from torch.distributed.rpc import RRef, rpc_async, remote
from time import time, sleep
import argparse
import sys
import json
import threading

import garfieldpp
from garfieldpp.worker import Worker
from garfieldpp.byzWorker import ByzWorker
from garfieldpp.server import Server
from garfieldpp.tools import get_bytes_com, convert_to_gbit, adjust_learning_rate

import aggregators
from math import log2, ceil

# The following fixes a `RuntimeError: received 0 items of ancdata` error, see:
#   https://github.com/pytorch/pytorch/issues/973#issuecomment-449756587
torch.multiprocessing.set_sharing_strategy("file_system")

import multiprocessing as mp
import asyncio

CIFAR_NUM_SAMPLES = 50000

import logging
import logging.config

logging_config = {
    "version": 1,
    "formatters": {
        "simple": {"format": "%(asctime)s [%(name)-10s] %(levelname)-10s %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        }
    },
    "loggers": {
        "garfieldpp.worker": {"level": "DEBUG", "handlers": ["console"], "propagate": False},
        "garfieldpp.server": {"level": "DEBUG", "handlers": ["console"], "propagate": False},
        "garfieldpp.datasets": {"level": "DEBUG", "handlers": ["console"], "propagate": False},
    },
    "root": {"level": "DEBUG", "handlers": ["console"]},
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)


def avg_agree(ps, gar, aggr_grad, num_iter, num_wait_ps, f):
    """Execute the average agreement protocol as in the paper
     Basically, exchange and aggregate gradients for log2(t) time
     Args
        ps		the local server object
        gar		GAR used for aggregation
        aggr_grad	the initial aggregated gradient
        num_iter	the number of iterations to be done; should be log2(t)
        num_wait_ps	the number of servers that should be waited for
        f               the number of byzantine servers
  """
    for _ in range(num_iter):
        ps.latest_aggr_grad = aggr_grad
        aggr_grads = ps.get_aggr_grads(num_wait_ps)
        aggr_grad = gar(gradients=aggr_grads, f=f)
    return aggr_grad


def node(
    rank,
    is_byzantine,
    world_size,
    batch,
    model,
    dataset,
    loss,
    num_iter,
    n,
    f,
    gar,
    optimizer,
    opt_args,
    non_iid,
    q,
    port,
):
    logger.debug(f"**** SETUP AT NODE {rank} ***")
    logger.debug(f"Number of nodes: {n}")
    logger.debug(f"Number of declared Byzantine nodes: {f}")
    logger.debug(f"GAR: {gar}")
    logger.debug(f"Dataset: {dataset}")
    logger.debug(f"Model: {model}")
    logger.debug(f"Batch size: {batch}")
    logger.debug(f"Loss function: {loss}")
    logger.debug(f"Optimizer: {optimizer}")
    logger.debug(f"Optimizer Args: {opt_args}")
    logger.debug(f"Assume Non-iid data? {non_iid}")
    logger.debug(f"------------------------------------")

    lr = opt_args["lr"]
    gar = aggregators.gars.get(gar)

    torch.manual_seed(1234)  # For reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)  # For reproducibility

    # No branching here! All nodes are created equal: no PS and no workers
    # Basically, each node has one PS object and one worker object
    rpc.init_rpc(
        "node:{}".format(rank),
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(
            init_method=f"tcp://localhost:{port}", rpc_timeout=10
        ),
    )
    logger.debug("RPC initialized")

    # rpc._set_rpc_timeout(100000)
    # initialize a worker here...the worker is created first because the server relies on the worker creation
    if is_byzantine:
        ByzWorker(rank, world_size, n, batch, model, dataset, loss, "random", f)
        logger.debug("Byzantine Worker created")
    else:
        Worker(rank, world_size, n, batch, model, dataset, loss)
        logger.debug("Worker created")

    # Initialize a parameter server
    ps = Server(
        rank,
        world_size,
        n,
        n,
        f,
        f,
        "node:",
        "node:",
        batch,
        model,
        dataset,
        optimizer,
        **opt_args,
    )
    logger.debug("Server created")

    sleep(5)  # works as a synchronization step
    logger.debug("Sleep done")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        ps.optimizer, milestones=[150, 250, 350], gamma=0.1
    )  # This line shows sophisticated stuff that can be done out of the Garfield++ library
    start_time = time()
    iter_per_epoch = CIFAR_NUM_SAMPLES // (
        n * batch
    )  # this value records how many iteration per sample
    logger.debug("One EPOCH consists of {} iterations".format(iter_per_epoch))
    sys.stdout.flush()

    # Training loop
    for i in range(num_iter):
        if (i + 1) % 10 == 0:
            logger.debug(f"[{rank}@{port}]: {i + 1}")
        if (
            i % (iter_per_epoch * 30) == 0 and i != 0
        ):  # One hack for better convergence with Cifar10
            lr *= 0.2
            adjust_learning_rate(ps.optimizer, lr)

        # Training step
        gradients = ps.get_gradients(i, n - f)  # get_gradients(iter_num, num_wait_wrk)
        # Aggregating gradients once is good for IID data; for non-IID, one should execute this log2(i)
        aggr_grad = gar(gradients=gradients, f=f)
        if non_iid:
            aggr_grad = avg_agree(ps, gar, aggr_grad, ceil(log2(i + 1)), n - f, f)
        ps.update_model(aggr_grad)

        # Then, communicate models, aggregate, and write the new model
        models = ps.get_models(n - f)
        aggr_models = gar(gradients=models, f=f)
        ps.write_model(aggr_models)
        ps.model.to("cpu:0")

        if (i + 1) % iter_per_epoch == 0:
            # Test step
            acc = ps.compute_accuracy()
            num_epochs = (i + 1) / iter_per_epoch
            logger.debug(
                "Epoch: {} Accuracy: {} Time: {}".format(
                    num_epochs, acc, time() - start_time
                )
            )
            sys.stdout.flush()

    final_acc = ps.compute_accuracy()

    rpc.shutdown()

    q.put(final_acc)


def training_run(n, f, port):
    assert f * 2 < n

    logger.info(f">>> Training run start: n={n} ; f={f}")

    q = mp.Queue()

    ps = []
    for rank in range(n):
        logger.info(f"Starting process with rank {rank}")
        p = mp.Process(
            target=node,
            kwargs=dict(
                rank=rank,
                is_byzantine=(rank < f),
                world_size=n,
                batch=125,
                model="convnet",
                dataset="mnist",
                loss="cross-entropy",
                num_iter=200,
                n=n,
                f=f,
                gar=("average" if f == 0 else "median"),
                optimizer="sgd",
                opt_args={"lr": 0.2, "momentum": 0.9, "weight_decay": 0.0005},
                non_iid=False,
                q=q,
                port=port,
            ),
        )
        p.start()
        ps.append(p)

    logger.info("Waiting for results")
    acc = [q.get(timeout=15*60) for _ in ps]

    for p in ps:
        p.join()

    logger.info(f"Final accuracies: {acc}")

    logger.info("<<< Training run end")

    return acc


async def main():
    loop = asyncio.get_running_loop()

    result = await asyncio.gather(
            loop.run_in_executor(None, training_run, 2, 0, 29700),
            loop.run_in_executor(None, training_run, 4, 1, 29701),
            )

    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
