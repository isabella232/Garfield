"""
Garfield Demonstrator
"""
import asyncio
from math import log2, ceil
import multiprocessing as mp
import queue
import threading
from time import time, sleep

import logging
import logging.config

import torch
import torch.nn.functional as F
import torch.distributed.rpc as rpc

from garfieldpp.worker import Worker
from garfieldpp.byzWorker import ByzWorker
from garfieldpp.server import Server
from garfieldpp.datasets import DatasetManager

import aggregators

from quart import (
    Quart,
    request,
    render_template,
    make_response,
)

# The following fixes a `RuntimeError: received 0 items of ancdata` error, see:
#   https://github.com/pytorch/pytorch/issues/973#issuecomment-449756587
torch.multiprocessing.set_sharing_strategy("file_system")


CIFAR_NUM_SAMPLES = 50000
NB_SAMPLES_PER_NODE = 100

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
        "garfieldpp.worker": {
            "level": "DEBUG",
            "handlers": ["console"],
            "propagate": False,
        },
        "garfieldpp.server": {
            "level": "DEBUG",
            "handlers": ["console"],
            "propagate": False,
        },
        "garfieldpp.datasets": {
            "level": "DEBUG",
            "handlers": ["console"],
            "propagate": False,
        },
    },
    "root": {"level": "DEBUG", "handlers": ["console"]},
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)


def avg_agree(ps, gar, aggr_grad, num_iter, num_wait_ps, f):
    """Execute the average agreement protocol as in the paper
    Basically, exchange and aggregate gradients for log2(t) time
    Args
       ps              the local server object
       gar             GAR used for aggregation
       aggr_grad       the initial aggregated gradient
       num_iter        the number of iterations to be done; should be log2(t)
       num_wait_ps     the number of servers that should be waited for
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
    nb_epochs,
    n,
    f,
    gar,
    optimizer,
    opt_args,
    non_iid,
    q,
    port,
):
    logger.debug("**** SETUP AT NODE %s ***", rank)
    logger.debug("Number of nodes: %d", n)
    logger.debug("Number of declared Byzantine nodes: %d", f)
    logger.debug("GAR: %s", gar)
    logger.debug("Dataset: %s", dataset)
    logger.debug("Model: %s", model)
    logger.debug("Batch size: %s", batch)
    logger.debug("Loss function: %s", loss)
    logger.debug("Optimizer: %s", optimizer)
    logger.debug("Optimizer Args: %s", opt_args)
    logger.debug("Assume Non-iid data? %s", non_iid)
    logger.debug("------------------------------------")

    gar = aggregators.gars.get(gar)

    torch.manual_seed(1234)  # For reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)  # For reproducibility

    # No branching here! All nodes are created equal: no PS and no workers
    # Basically, each node has one PS object and one worker object
    rpc.init_rpc(
        f"node:{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(
            init_method=f"tcp://localhost:{port}"
        ),
    )
    logger.debug("RPC initialized")

    train_size = n * NB_SAMPLES_PER_NODE

    # rpc._set_rpc_timeout(100000)
    # initialize a worker here...the worker is created first because the server relies on the worker creation
    if is_byzantine:
        ByzWorker(
            rank,
            world_size,
            n,
            batch,
            model,
            dataset,
            loss,
            "random",
            f,
            train_size=train_size,
        )
        logger.debug("Byzantine Worker created")
    else:
        Worker(rank, world_size, n, batch, model, dataset, loss, train_size=train_size)
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
        train_size=train_size,
        **opt_args,
    )
    logger.debug("Server created")

    sleep(5)  # works as a synchronization step
    logger.debug("Sleep done")

    start_time = time()

    dataset_size = len(
        DatasetManager(dataset, 0, 0, 0, 0, train_size=train_size).fetch_dataset()
    )

    iter_per_epoch = ceil(
        dataset_size / (n * batch)
    )  # this value records how many iteration per sample
    logger.debug("One EPOCH consists of %d iterations", iter_per_epoch)
    num_iter = nb_epochs * iter_per_epoch
    logger.debug("Doing %d iterations to cover %d epochs", num_iter, nb_epochs)

    q.put({"rank": rank, "progress": 0})

    # Training loop
    for i in range(num_iter):
        ## if (
        ##     i % (iter_per_epoch * 30) == 0 and i != 0
        ## ):  # One hack for better convergence with Cifar10
        ##     lr *= 0.2
        ##     adjust_learning_rate(ps.optimizer, lr)

        # Training step
        gradients = ps.get_gradients(i, n)  # get_gradients(iter_num, num_wait_wrk)
        # Aggregating gradients once is good for IID data; for non-IID, one should execute this log2(i)
        aggr_grad = gar(gradients=gradients, f=f)
        if non_iid:
            aggr_grad = avg_agree(ps, gar, aggr_grad, ceil(log2(i + 1)), n - f, f)
        ps.update_model(aggr_grad)

        # Then, communicate models, aggregate, and write the new model
        models = ps.get_models(n)
        aggr_models = gar(gradients=models, f=f)
        ps.write_model(aggr_models)
        ps.model.to("cpu:0")

        if (i + 1) % iter_per_epoch == 0:
            # Test step
            acc = ps.compute_binary_accuracy()
            num_epochs = (i + 1) / iter_per_epoch
            logger.debug(
                "Epoch: %d Accuracy: %f Time: %d", num_epochs, acc, time() - start_time
            )

        if (i + 1) % 10 == 0:
            logger.debug("[%d@%d]: %d%%", rank, port, (i + 1) * 100 // num_iter)
        q.put({"rank": rank, "progress": (i + 1) * 100 // num_iter})

    final_acc = ps.compute_binary_accuracy()

    rpc.shutdown()

    q.put({"rank": rank, "progress": 100, "result": final_acc})


class Trainer:
    TIMEOUT_PROGRESS_SEC = 1 * 60
    TIMEOUT_TERMINATE_SEC = 10

    def __init__(self, n, f, gar, port):
        if n < 1 or n > 10:
            raise ValueError("The total number of nodes must be between 1 and 10")

        # if n <= 2 * f:
        #     raise ValueError("The total number of nodes must be > 2 * the number of byzantine nodes")

        self.n = n
        self.f = f
        self.gar = gar
        self.port = port

        self.lock = threading.Lock()
        self.status = None
        self.fut = None

    def train(self):
        logger.info(f">>> Training run start: n={self.n} ; f={self.f}")

        dataset = "pima"
        model = "pimanet"
        batch_size = 16
        nb_epochs = 15

        self.status = {rank: -1 for rank in range(self.n)}

        q = mp.Queue()

        ps = []
        for rank in range(self.n):
            logger.info("Starting process with rank %d", rank)
            p = mp.Process(
                target=node,
                kwargs=dict(
                    rank=rank,
                    is_byzantine=(rank < self.f),
                    world_size=self.n,
                    batch=batch_size,
                    model=model,
                    dataset=dataset,
                    loss="binary-cross-entropy",
                    nb_epochs=nb_epochs,
                    n=self.n,
                    f=self.f,
                    gar=self.gar,
                    optimizer="rmsprop",
                    opt_args={"lr": 0.001, "momentum": 0.9, "weight_decay": 0.0005},
                    non_iid=False,
                    q=q,
                    port=self.port,
                ),
            )
            p.start()
            ps.append(p)

        logger.info("Waiting for results")

        try:
            acc = []
            while len(acc) < len(ps):
                prog = q.get(timeout=self.TIMEOUT_PROGRESS_SEC)
                # logger.debug(f"Progress received: {prog}")

                with self.lock:
                    self.status[prog["rank"]] = prog["progress"]

                if "result" in prog:
                    acc.append(prog["result"])

        except queue.Empty as exc:
            # Try to cleanup
            for p in ps:
                p.kill()

            logger.exception("Timeout occurred while waiting for progress")

            raise Exception("Timeout while waiting for progress") from exc

        for p in ps:
            p.join(timeout=self.TIMEOUT_TERMINATE_SEC)
            if p.exitcode is None:
                logger.warning("Process %s did not terminate", p)

        logger.info("Final accuracies: %s", acc)

        logger.info("<<< Training run end")

        return acc

    def run(self):
        loop = asyncio.get_event_loop()
        self.fut = loop.run_in_executor(None, self.train)

    def get_status(self):
        if self.fut.done():
            try:
                res = self.fut.result()
                status = {"result": sum(res) / len(res)}
            except Exception as exc:
                status = {"error": str(exc)}
        else:
            with self.lock:
                status = {"progress": sum(self.status.values()) // len(self.status)}

        logger.debug("Returning status: %s", status)

        return status


class PortManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.port = 29700

    def get_next_port(self):
        with self.lock:
            port = self.port
            self.port += 1

        logger.debug("Returning port: %d", port)
        return port


class Trainers:
    def __init__(self):
        self.lock = threading.Lock()
        self.trainers = {}
        self.id = 0

    def submit(self, trainer):
        with self.lock:
            trainer_id = self.id
            self.id += 1

            self.trainers[trainer_id] = trainer

        trainer.run()

        return trainer_id

    def get_status(self, trainer_id):
        with self.lock:
            trainer = self.trainers.get(trainer_id)

        if trainer is None:
            raise ValueError(f"Trainer ID {trainer_id} does not exist")

        return trainer.get_status()


app = Quart(__name__)


@app.route("/", methods=["GET"])
async def index():
    return await render_template("index.html")


@app.route("/", methods=["POST"])
async def train():
    form = await request.get_json()

    try:
        n = int(form["n"])
        f = int(form["f"])
        gar = form.get("gar", "average")

        trainer = Trainer(n, f, gar, pm.get_next_port())
        trainer_id = trainers.submit(trainer)

    except Exception as e:
        logger.error(e)
        return await make_response({"error": str(e)}, 400)

    return {"trainerId": trainer_id}


@app.route("/status", methods=["GET"])
async def get_status():
    args = request.args

    try:
        trainer_id = int(args["trainer_id"])
        status = trainers.get_status(trainer_id)

    except Exception as e:
        logger.error(e)
        return await make_response({"error": str(e)}, 400)

    return status


@app.cli.command("init_demo")
def init_demo():
    from garfieldpp.datasets import DatasetManager

    # Pre-load the Pima Indians Diabetes dataset, and build the C++ aggretors as a side-effect
    dm = DatasetManager("pima", 0, 0, 0, 0)
    dm.fetch_dataset()


pm = PortManager()
trainers = Trainers()

if __name__ == "__main__":
    app.run()
