FROM debian:9-slim

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt update -qq && \
    apt install -qq -y curl git python3 python3-pip python3-setuptools python3-six && \
    \
    mkdir -p /root/.noisemaker/fonts && \
    cd /root/.noisemaker/fonts && \
    /usr/bin/curl -s https://releases.pagure.org/liberation-fonts/liberation-fonts-ttf-2.00.1.tar.gz \
        | /bin/tar xz --strip-components=1 && \
    cd / && \
    \
    pip3 install -q git+https://github.com/aayars/py-noisemaker && \
    pip3 install -q tensorflow==1.3.0 && \
    \
    apt install -qq -y cpanminus gcc libjpeg-dev libpng-dev make perl-doc && \
    PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install Math::Fractal::Noisemaker' && \
    apt remove -qq --purge -y cpanminus gcc make && \
    \
    apt remove -qq --purge -y curl git python3-pip && \
    apt autoremove -qq -y && \
    \
    noisemaker --help > /dev/null

CMD noisemaker --help
