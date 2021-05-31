FROM python:3.9-slim

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt update -qq && \
    apt install -qq -y curl git python3-pip python3-setuptools python3-six ffmpeg imagemagick libopenexr-dev zlib1g-dev && \
    \
    mkdir -p /root/.noisemaker/fonts && \
    cd /root/.noisemaker/fonts && \
    /usr/bin/curl -s https://s3.wasabisys.com/noisemakerbot-assets-east/fonts/liberation-fonts-ttf-2.00.1.tar.gz \
        | /bin/tar xz --strip-components=1 && \
    /usr/bin/curl --output Jura-Regular.ttf https://s3.wasabisys.com/noisemakerbot-assets-east/fonts/Jura-Regular.ttf && \
    cd / && \
    \
    pip3 install -q git+https://github.com/aayars/py-noisemaker && \
    pip3 install -q tensorflow==2.5.0 && \
    \
    apt remove -qq --purge -y curl git python3-pip && \
    apt autoremove -qq -y && \
    \
    noisemaker --help > /dev/null

CMD noisemaker --help
