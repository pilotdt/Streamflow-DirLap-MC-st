FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
RUN apt-get update && apt-get install -y git
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip setuptools wheel
RUN pip install hydra-core --upgrade
RUN pip install -r requirements.txt
RUN pip install mamba-ssm[causal-conv1d] --no-build-isolation


