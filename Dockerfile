FROM ubuntu:16.04
# python3.7
RUN apt-get update
RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt install -y python3.7
RUN apt-get install -y python3-pip
RUN apt install git -y

RUN apt-get update && \
    apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-dev

COPY ./src /mlapi/src
COPY ./model /mlapi/model
COPY requirements.txt /mlapi/src/requirements.txt
WORKDIR /mlapi/src

# Install production dependencies.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 7000
