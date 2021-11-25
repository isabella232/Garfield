# Garfield Demonstrator

## Introduction

This is a demonstrator of the Garfield framework, based on the LEARN
decentralized learning application.).

It is part of the [C4DT Factory
demonstrators](https://factory.c4dt.org/showcase/#dropdown=product_demo), and
can be viewed by clicking
[here](https://factory.c4dt.org/incubator/garfield/demo/).

## Architecture

The demonstrator runs as a web application, implemented in Python3 using
[Quart](https://pypi.org/project/quart/) for the back-end and
[Vue.js](https://vuejs.org/), along with [its Material
widgets](https://www.creative-tim.com/vuematerial/), for the front-end.

## How to build

A `Dockerfile` is provided to build a container:

```
$ docker build --file pytorch_impl/Dockerfile --tag garfield-demo .
```

When run, the container exposes the application on its port 8000:

```
$ docker run --name garfield-demo --rm garfield-demo

[in another terminal]
$ docker exec garfield-demo cat /etc/hosts
...
```

read the container IP address from the last line, and point your browser to
`<IP_ADDR>:8000` .

## How to run locally for development

### Requirements

- Python 3.7

### Procedure

All commands are run from the top directory in the repository.

- Create a Python virtual environment:

```
$ python3.7 -m venv demo_venv
$ . ./demo_venv/bin/activate
```

- Install the Python dependencies:

```
$ pip install -r pytorch_impl/requirements.txt
```

- Run the application:

```
$ QUART_APP=pytorch_impl/applications/LEARN/demo:app quart run
```

The first time it is run, the system will compile the native imnplementations
of the aggretators, which can take a few minutes.

Eventually, the console will show the following:

```
[...] Running on http://127.0.0.1:5000 (CTRL + C to quit)
```

Point your browser to the above address, and you should see the demo.

The demo is contained in:

- `pytorch_impl/applications/LEARN/demo.py` for the back-end
- `pytorch_impl/applications/LEARN/templates/index.html` for the front-end

If you edit and save the back-end, Quart will automatically detect the changes
and restart with the new code. The same is however not true for the front-end,
so when you change it you will need to restart quart (CTRL-C and rerun).
