#!/bin/sh

container=tstreamer
DOCKER_TAG="tstreamer/tstreamer:latest-xaser-gpu"

echo "Stopping $container"
docker stop $container
echo "Removing $container"
docker rm $container
echo "Running $container"
docker run -ti -d --runtime=nvidia --ipc=host --net="host" --restart=always --name $container -v torchserve_bash:/home/bash -v /mnt/nas_downloads:/mnt/nas_downloads $DOCKER_TAG