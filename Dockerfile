FROM tensorflow/tensorflow:1.15.5-gpu-py3-jupyter
ARG GIT_USEREMAIL
ARG GIT_USERNAME

WORKDIR /usr/src/
COPY requirements.txt ./

RUN apt-get update && apt upgrade -y && \
    apt-get install git -y && \
    git config --global user.email $GIT_USEREMAIL && git config --global user.name $GIT_USERNAME

COPY . /usr/src/

RUN pip install -U pip setuptools && \
    pip install -U --no-cache-dir -r requirements.txt 

RUN cd attention-ocr && python setup.py install

# ENTRYPOINT [ "/bin/bash" ]