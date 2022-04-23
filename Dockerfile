FROM tensorflow/tensorflow:1.15.5-gpu
ARG GIT_USEREMAIL
ARG GIT_USERNAME

WORKDIR /usr/src/

RUN apt-get update && apt upgrade -y && \
    apt-get install git -y && \
    git config --global user.email $GIT_USEREMAIL && git config --global user.name $GIT_USERNAME

RUN : \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.7-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :

RUN python3.7 -m venv /venv
ENV PATH=/venv/bin:$PATH

COPY . /usr/src/

RUN pip install -U pip setuptools && \
    pip install -U --no-cache-dir -r requirements.txt 

RUN cd attention-ocr && python setup.py install

# ENTRYPOINT [ "/bin/bash" ]