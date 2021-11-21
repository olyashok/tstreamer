#!/bin/sh

container=tstreamer_cropper
DOCKER_TAG="tstreamer/tstreamer:latest-xaser-gpu"

echo "Stopping $container"
docker stop $container
echo "Removing $container"
docker rm $container
echo "Running $container"
docker run -ti -d --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility --ipc=host --net="host" --restart=always --name $container -v torchserve_bash:/home/bash -v /mnt/nas_downloads:/mnt/nas_downloads $DOCKER_TAG /mnt/nas_downloads/deepstack/tstreamer/tstreamer/crop.py
sleep 5
