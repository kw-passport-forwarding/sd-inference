FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN apt update && apt install curl vim -y
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./ /app