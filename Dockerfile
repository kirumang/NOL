FROM nvidia/cudagl:10.1-devel-ubuntu18.04
MAINTAINER Kiru Park (park@acin.tuwien.ac.at)
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get update && apt-get install -y \
	python \
	python-pip \
	git \
	vim \
	wget \
        cmake
 
