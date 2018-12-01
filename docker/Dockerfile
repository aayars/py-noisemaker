FROM debian:9-slim

RUN apt update && \
    apt install -y git python3 python3-pip python3-setuptools python3-six && \
    \
    pip3 install git+https://github.com/aayars/py-noisemaker && \
    \
    apt remove -y git python3-pip && \
    apt autoremove -y

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD noisemaker --help
