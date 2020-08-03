FROM nvidia/cudagl:10.1-devel-ubuntu18.04
MAINTAINER Kiru Park (park@acin.tuwien.ac.at)
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get update && apt-get install -y \
	python3 \
	python3-pip \
	git \
	vim \
	wget \
	cmake\
	libsm6\
	libxrender1\
	libcudnn7=7.6.4.38-1+cuda10.1  \
	libcudnn7-dev=7.6.4.38-1+cuda10.1

RUN mkdir -p /root/src
COPY . /root/src/

#Install python requirements
RUN pip3 install -U pip
RUN cd /root/src/ && pip3 install -r requirements.txt

#Build dirt
RUN cd /root/src/dirt && mkdir build && cd build && cmake ../csrc && make && cd .. && pip3 install -e .
