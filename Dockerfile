FROM tensorflow/tensorflow:1.4.0-rc0-devel-gpu

RUN apt-get update && \
    apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

COPY ./src /mlapi/src
COPY ./model /mlapi/model
COPY requirements.txt /mlapi/src/requirements.txt
WORKDIR /mlapi/src

# Install production dependencies.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 7000
