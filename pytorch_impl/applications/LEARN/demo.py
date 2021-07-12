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
from garfieldpp.server import Server
from garfieldpp.tools import get_bytes_com, convert_to_gbit, adjust_learning_rate

import aggregators
from math import log2, ceil

# The following fixes a `RuntimeError: received 0 items of ancdata` error, see:
#   https://github.com/pytorch/pytorch/issues/973#issuecomment-449756587
torch.multiprocessing.set_sharing_strategy("file_system")

CIFAR_NUM_SAMPLES = 50000

import logging

logging.basicConfig(format="%(asctime)s [%(name)-10s] %(levelname)-10s %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


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
):
    log.debug(f"**** SETUP AT NODE {rank} ***")
    log.debug(f"Number of nodes: {n}")
    log.debug(f"Number of declared Byzantine nodes: {f}")
    log.debug(f"GAR: {gar}")
    log.debug(f"Dataset: {dataset}")
    log.debug(f"Model: {model}")
    log.debug(f"Batch size: {batch}")
    log.debug(f"Loss function: {loss}")
    log.debug(f"Optimizer: {optimizer}")
    log.debug(f"Optimizer Args: {opt_args}")
    log.debug(f"Assume Non-iid data? {non_iid}")
    log.debug(f"------------------------------------")

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
            init_method="tcp://localhost:29700", rpc_timeout=10
        ),
    )
    log.debug("RPC initialized")

    # rpc._set_rpc_timeout(100000)
    # initialize a worker here...the worker is created first because the server relies on the worker creation
    Worker(rank, world_size, n, batch, model, dataset, loss)
    log.debug("Worker created")

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
    log.debug("Server created")

    sleep(5)  # works as a synchronization step
    log.debug("Sleep done")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        ps.optimizer, milestones=[150, 250, 350], gamma=0.1
    )  # This line shows sophisticated stuff that can be done out of the Garfield++ library
    start_time = time()
    iter_per_epoch = CIFAR_NUM_SAMPLES // (
        n * batch
    )  # this value records how many iteration per sample
    log.debug("One EPOCH consists of {} iterations".format(iter_per_epoch))
    sys.stdout.flush()

    accuracies = []

    # Training loop
    for i in range(num_iter):
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
            log.debug(
                "Epoch: {} Accuracy: {} Time: {}".format(
                    num_epochs, acc, time() - start_time
                )
            )
            sys.stdout.flush()

            accuracies.append(acc)

    rpc.shutdown()

    q.put(accuracies[-1])


def main():
    n = 2
    f = 0
    assert f * 2 < n

    import multiprocessing as mp

    log.info("Demo start")

    q = mp.Queue()

    ps = []
    for rank in range(n):
        log.info(f"Starting process with rank {rank}")
        p = mp.Process(
            target=node,
            kwargs=dict(
                rank=rank,
                world_size=n,
                batch=125,
                model="convnet",
                dataset="mnist",
                loss="cross-entropy",
                num_iter=200,
                n=n,
                f=f,
                gar="average",
                optimizer="sgd",
                opt_args={"lr": 0.2, "momentum": 0.9, "weight_decay": 0.0005},
                non_iid=False,
                q=q,
            ),
        )
        p.start()
        ps.append(p)

    log.info("Waiting for results")
    acc = [q.get(timeout=90) for _ in ps]

    for p in ps:
        p.join()

    log.info(f"Final accuracies: {acc}")


if __name__ == "__main__":
    main()
