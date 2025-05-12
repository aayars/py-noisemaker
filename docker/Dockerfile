FROM python:3.13-slim

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt update -qq && \
    apt install -qq -y pkg-config libhdf5-dev gcc git ffmpeg imagemagick && \
    pip install -q tensorflow==2.16.1 && \
    pip install -q git+https://github.com/aayars/py-noisemaker && \
    apt remove -qq --purge -y pkg-config gcc git && \
    apt autoremove -qq -y && \
    noisemaker --help > /dev/null

CMD ["noisemaker", "--help"]
