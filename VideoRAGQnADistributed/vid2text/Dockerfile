ARG OS_VER=22.04
ARG OS_IMAGE=ubuntu

FROM ${OS_IMAGE}:${OS_VER}
USER root
WORKDIR /vid2text

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends \
    python3.9 python3.9-venv python3.9-dev gcc build-essential curl vim libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# avoid installing packages while re-build
COPY ./vid2text/requirements.txt /vid2text/requirements.txt

RUN python3.9 -m venv /root/vid2text
RUN . /root/vid2text/bin/activate && \
    pip install --no-cache-dir -r requirements.txt && \
    deactivate

RUN . /root/vid2text/bin/activate && \
    pip install datetime && deactivate

ENV PATH=/root/vid2text/bin:$PATH

COPY ./vid2text /vid2text/
COPY ./conf /vid2text/


ENTRYPOINT ["bash","start.sh"]