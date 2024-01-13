#!/bin/bash

DOCKER_TAG="tstreamer2/tstreamer2:latest"
NVIDIA_DRIVER_VERSION=`modinfo nvidia | grep "^version:" | awk '{print $2}'`

DOCKER_BUILDKIT=1 docker build --file Dockerfile -t $DOCKER_TAG .
