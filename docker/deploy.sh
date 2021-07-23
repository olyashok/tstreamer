#!/bin/sh

usage="$(basename "$0") [-d msdelay] -r stream -n name -t directory_to_save [-d delay:5000] [-l loglevel|DEBUG] [-f saveframe|False]"

if [ $# -eq 0 ]; then
	echo "$usage"
	exit
fi

delay=5000
loglevel=DEBUG
frame=False
tess=False

while getopts ':r:n:d:t:l:f:k:' option; do
  case "$option" in
    n) name=$OPTARG
       ;;
    r) rtsp=$OPTARG
       ;;
    d) delay=$OPTARG
        ;;
    t) target_dir=$OPTARG
        ;;
    l) loglevel=$OPTARG
        ;;
    k) tess=$OPTARG
        ;;
    f) frame=$OPTARG
        ;;
    :) printf "missing argument for -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
   \?) printf "illegal option: -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
  esac
done
shift $((OPTIND - 1))

container=tstreamer_${name}
DOCKER_TAG="tstreamer/tstreamer:latest-xaser-gpu"

echo "Stopping $container"
docker stop $container
echo "Removing $container"
docker rm $container
echo "Running $container"
docker run -ti -d --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility --ipc=host --net="host" --restart=always --name $container -v torchserve_bash:/home/bash -v /mnt/nas_downloads:/mnt/nas_downloads $DOCKER_TAG /usr/bin/python3 /mnt/nas_downloads/deepstack/tstreamer/tstreamer/gs2mqtt.py --name=${name} --stream=${rtsp} --torchserve-ip=192.168.10.23 --directory="${target_dir}" --save-timestamped=True --save-latest=True --save-crops=True --show-boxes=True --save-labels=True --mqtt-ip=192.168.10.22 --mqtt-user=xaser --mqtt-password=SnetBil8a --debug=$loglevel --save-frame=$frame --delay=${delay} --fire-events=True --read-time=${tess} --detect-dups=60