FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel
#FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel
 
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
 
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update
 
# Install all apt-get packages. Merendeel nodig voor anaconda
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 ffmpeg -y \
    git mercurial subversion libbz2-dev libz-dev libpng-dev
 
# Anaconda installeren
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

RUN conda info

RUN conda install nodejs
RUN conda install -c conda-forge jupyterlab
RUN conda install -c conda-forge jupytext

RUN pip install --upgrade pip
RUN pip install opencv-python scikit-image pycocotools
RUN pip install --force-reinstall --no-deps h5py
RUN pip install pandas seaborn numpy sacred Pillow
RUN pip install matplotlib

RUN jupyter labextension install jupyterlab-jupytext
RUN jupyter labextension install @jupyterlab/git
RUN pip install --upgrade jupyterlab-git
RUN jupyter server extension enable --py jupyter_http_over_ws

#RUN jupyter serverextension enable --py jupyterlab_git

# Install ssh server, used by PyCharm for remote execution
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/^#?PermitUserEnvironment\s+.*/PermitUserEnvironment yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config
RUN mkdir /root/.ssh
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
EXPOSE 22

RUN pip install ftfy regex tqdm
RUN pip install -U scikit-learn scipy
RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install lightning

RUN pip install wandb
RUN pip install iopath timm==0.6.7 einops
RUN pip install faiss-cpu

RUN apt-get update && apt-get install -y tmux
RUN apt-get install -y yacs python-dotenv
#ENTRYPOINT service ssh restart && /bin/bash
#ENTRYPOINT [ "/usr/bin/tini", "--" ]
#CMD [ "/bin/bash" ]
 
# Do not detach
# ENTRYPOINT /usr/sbin/sshd -D
