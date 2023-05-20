#!/bin/bash

cropdir="/mnt/nas_downloads/data/hassio/tstreamer"
hostname=`hostname`

if [ "$hostname" = "xaser-ryzen" ]; then
    cropdir="/mnt/localshared/data/hassio/tstreamer"
fi

if [ ! -d "$cropdir" ]; then
    echo "$cropdir is not mounted. exiting."
    exit 1
fi

cd "$cropdir"

min=1
st=2
lt=14

find "$cropdir" -type f -regex ".*_nobox_.*\.jpg" -mtime +$min -delete
find "$cropdir" -type f -regex ".*_yolov[0-9]*[a-z]*_.*\.jpg"  -mtime +$min -delete
find "$cropdir" -type f -regex ".*_box_.*\.jpg" -mtime +$st -delete
find "$cropdir" -type f -regex ".*_unifi[a-z0-9]*_.*\.jpg"  -mtime +$lt -delete
