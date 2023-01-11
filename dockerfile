FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
# RUN git clone https://github.com/Hyu-Zhang/VQA.git
# RUN gdown https://drive.google.com/uc?id=1fxrVI-auCQ118ldfxrbf35UnyvYhSQ3G
# RUN unzip checkpoints.zip
COPY VQA/ /usr/local/VQA/

RUN apt-get update
RUN apt-get install ffmpeg curl git gcc g++ gnupg2 libgl1-mesa-glx libsm6 libxext6 libglib2.0-dev -y

RUN python -m pip install pip==21.2.4
RUN pip install gdown wget -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN export CFLAGS='-std=c++11'

WORKDIR /usr/local/VQA/

RUN pip install -r ./requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /usr/local/VQA/fairseq

RUN pip install ./

WORKDIR /usr/local/VQA/
ENTRYPOINT ["bash", "run.sh"]
