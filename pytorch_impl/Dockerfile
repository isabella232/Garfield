FROM python:3.7.10-slim-buster

COPY pytorch_impl/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && \
    apt-get install --no-install-recommends --yes \
    ninja-build g++

COPY . ./Garfield

RUN QUART_APP=Garfield/pytorch_impl/applications/LEARN/demo:app quart init_demo

CMD [ "hypercorn", \
        "--bind", "0.0.0.0:8000", \
        "Garfield/pytorch_impl/applications/LEARN/demo:app"]

