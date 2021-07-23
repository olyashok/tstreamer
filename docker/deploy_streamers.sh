#!/bin/sh

#approach
./deploy.sh -r http://192.168.10.184/snap.jpeg -n uvc_g4_pro_946e -d 2000 -t /mnt/nas_downloads/data/hassio/tstreamer -l DEBUG -f True -k True
#doorbell
./deploy.sh -r http://192.168.10.217/snap.jpeg -n uvc_g4_doorbell -d 3000 -t /mnt/nas_downloads/data/hassio/tstreamer -l DEBUG -f True -k True
#driveway
./deploy.sh -r http://192.168.10.246/snap.jpeg -n uvc_g3_pro_a -d 500 -t /mnt/nas_downloads/data/hassio/tstreamer -l DEBUG -f True -k True
#dock
./deploy.sh -r http://192.168.10.198/snap.jpeg -n uvc_g3_pro_c -d 10000 -t /mnt/nas_downloads/data/hassio/tstreamer -l DEBUG -f True -k True

#./deploy.sh -r rtsp://192.168.10.1:7447/JdCDi8nwunLfXqbD -n uvc_g3_flex_b -d 10 -t /mnt/nas_downloads/data/hassio/tstreamer -l INFO -f True -k True
#./deploy.sh -r rtsp://192.168.10.1:7447/zEVDUcica28HNKVu -n uvc_g3_instant_1
