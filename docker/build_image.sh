#!/bin/bash

BASE_IMAGE="nvcr.io/nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu18.04"
DOCKER_TAG="tstreamer/tstreamer:latest-xaser-gpu"

DOCKER_BUILDKIT=1 docker build --file Dockerfile --no-cache --build-arg BASE_IMAGE=$BASE_IMAGE -t $DOCKER_TAG .
