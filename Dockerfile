FROM huggingface/transformers-pytorch-gpu
ARG GIT_USEREMAIL
ARG GIT_USERNAME

WORKDIR /usr/src/

RUN git config --global user.email $GIT_USEREMAIL && git config --global user.name $GIT_USERNAME

COPY requirements.txt ./

RUN apt-get update && apt upgrade -y && \
    apt-get install -y libsm6 libxrender1 libfontconfig1 libxext6 libgl1-mesa-glx && \
    pip install -U pip setuptools && \
    pip install -U --no-cache-dir -r requirements.txt 

ENV PYTHONPATH="/usr/src/"

COPY . /usr/src/

ENTRYPOINT [ "/bin/sh" ]
# CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--NotebookApp.token=''", "--allow-root"]