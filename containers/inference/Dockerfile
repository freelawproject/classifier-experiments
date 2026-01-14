FROM runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2204

WORKDIR /

RUN apt-get update && \
    apt-get install -y git gcc g++ make libmagic-dev && \
    rm -rf /var/lib/apt/lists/*
RUN pip install flash-attn --no-build-isolation
RUN pip install runpod
RUN pip install git+https://github.com/freelawproject/classifier-experiments

ADD handler.py .

CMD [ "python", "-u", "/handler.py" ]