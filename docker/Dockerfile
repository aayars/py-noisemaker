FROM debian:9-slim

RUN apt update && \
    apt install -y curl git python3 python3-pip python3-setuptools python3-six && \
    \
    mkdir -p /root/.noisemaker/fonts && \
    cd /root/.noisemaker/fonts && \
    /usr/bin/curl https://releases.pagure.org/liberation-fonts/liberation-fonts-ttf-2.00.1.tar.gz \
        | /bin/tar xz --strip-components=1 && \
    cd / \
    \
    pip3 install git+https://github.com/aayars/py-noisemaker && \
    pip3 install tensorflow==1.3.0 && \
    \
    apt remove -y curl git python3-pip && \
    apt autoremove -y

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD noisemaker --help
