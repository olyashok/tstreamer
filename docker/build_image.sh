#!/bin/bash

BASE_IMAGE="nvcr.io/nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04"
DOCKER_TAG="tstreamer/tstreamer:latest-xaser-gpu"
NVIDIA_DRIVER_VERSION=`modinfo nvidia | grep "^version:" | awk '{print $2}'`

DOCKER_BUILDKIT=1 docker build --file Dockerfile --build-arg BASE_IMAGE=$BASE_IMAGE --build-arg NVIDIA_DRIVER_VERSION="$NVIDIA_DRIVER_VERSION" -t $DOCKER_TAG .
