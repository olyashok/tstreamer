#!/bin/sh

name=$2
stream=$1
delay=$3

container=tstreamer_${name}
DOCKER_TAG="tstreamer2/tstreamer2:latest"

echo "Stopping $container"
docker stop $container
echo "Removing $container"
docker rm $container
echo "Running $container"
docker run -ti -d \
--ipc=host --net="host" \
--restart=always --name $container \
-v /mnt/nas_downloads/deepstack/tstreamer/app:/app \
-v /mnt/nas_downloads:/mnt/nas_downloads \
-v /mnt/localshared:/mnt/localshared \
$DOCKER_TAG \
/usr/bin/python3 /app/stream2mqtt.py --name=${name} --stream=${stream} --delay=${delay} --debug=DEBUG