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
RUN python3.7 -m pip install pip
RUN python3.7 -m pip install pip --upgrade pip
RUN python3.7 -m pip install -r /mlapi/src/requirements.txt

EXPOSE 7000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7000"]