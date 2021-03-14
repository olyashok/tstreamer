#!/bin/bash

cd /mnt/nas_downloads/data/gstest

rm /mnt/nas_downloads/data/gstest/playlist.m3u8
rm /mnt/nas_downloads/data/gstest/*.ts

gst-launch-1.0 -e \
videomixer name=mix \
        sink_0::xpos=0   sink_0::ypos=0  sink_0::alpha=0 \
        sink_1::xpos=0   sink_1::ypos=0 \
        sink_2::xpos=640 sink_2::ypos=0 \
        sink_3::xpos=0   sink_3::ypos=360 \
        sink_4::xpos=640 sink_4::ypos=360 \
    ! videoconvert ! clockoverlay ! nvh264enc ! mpegtsmux ! hlssink playlist-root=http://192.168.10.23:10000 location=/mnt/nas_downloads/data/gstest/segment_%05d.ts target-duration=5 max-files=5 \
videotestsrc pattern="black" \
    ! video/x-raw,format=AYUV,width=1280,height=720 \
    ! mix.sink_0 \
rtspsrc location="rtsp://192.168.10.1:7447/fZSTd3E7sQ7AWt7Q" \
    ! rtph264depay ! h264parse ! nvh264dec \
    ! video/x-raw,width=640,height=360 \
    ! mix.sink_1 \
rtspsrc location="rtsp://192.168.10.1:7447/aVw23e6tyMAWliBD" \
    ! rtph264depay ! h264parse ! nvh264dec \
    ! video/x-raw,width=640,height=360 \
    ! mix.sink_2 \
rtspsrc location="rtsp://192.168.10.1:7447/fZSTd3E7sQ7AWt7Q" \
    ! rtph264depay ! h264parse ! nvh264dec \
    ! video/x-raw,width=640,height=360 \
    ! mix.sink_3 \
rtspsrc location="rtsp://192.168.10.1:7447/fZSTd3E7sQ7AWt7Q" \
    ! rtph264depay ! h264parse ! nvh264dec \
    ! video/x-raw,width=640,height=360 \
    ! mix.sink_4