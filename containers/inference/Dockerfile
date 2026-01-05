FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /

RUN apt-get update && \
    apt-get install -y git gcc g++ make libmagic-dev && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get remove python3-blinker -y
RUN pip install flash-attn --no-build-isolation
RUN pip install runpod
RUN pip install git+https://github.com/freelawproject/classifier-experiments

ADD handler.py .

CMD [ "python", "-u", "/handler.py" ]