#!/bin/sh

container=tstreamer
DOCKER_TAG="tstreamer/tstreamer_triton:latest-xaser-gpu"

echo "Stopping $container"
docker stop $container
echo "Removing $container"
docker rm $container
echo "Running $container"
docker run -ti -d --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility --ipc=host --net="host" -p 35555:35555 --restart=always --name $container -v torchserve_bash:/home/bash -v /mnt/nas_downloads:/mnt/nas_downloads -v /mnt/localshared:/mnt/localshared $DOCKER_TAG
sleep 5
#docker exec -it $container pip install pytesseract