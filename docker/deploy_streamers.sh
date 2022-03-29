#!/bin/sh

#approach
./deploy.sh -r http://192.168.10.184/snap.jpeg -n uvc_g4_pro_946e_high -d 5000 -t /mnt/nas_downloads/data/hassio/tstreamer -l DEBUG -f True -k True -u 60
#doorbell
./deploy.sh -r http://192.168.10.217/snap.jpeg -n uvc_g4_doorbell_high -d 3000 -t /mnt/nas_downloads/data/hassio/tstreamer -l DEBUG -f True -k True -u 0
#driveway
./deploy.sh -r http://192.168.10.246/snap.jpeg -n uvc_g3_pro_a_high -d 1000 -t /mnt/nas_downloads/data/hassio/tstreamer -l DEBUG -f True -k True -u 0
#dock
./deploy.sh -r http://192.168.10.198/snap.jpeg -n uvc_g3_pro_c_high -d 10000 -t /mnt/nas_downloads/data/hassio/tstreamer -l DEBUG -f True -k True -u 0
#pool
./deploy.sh -r http://192.168.10.214/snap.jpeg -n uvc_g3_pro_b_high -d 5000 -t /mnt/nas_downloads/data/hassio/tstreamer -l DEBUG -f True -k True -u 60
#garage
./deploy.sh -r http://192.168.10.193/snap.jpeg -n uvc_g3_flex_a_high -d 10000 -t /mnt/nas_downloads/data/hassio/tstreamer -l DEBUG -f True -k True -u 60
#deck
./deploy.sh -r http://192.168.10.191/snap.jpeg -n uvc_g4_bullet_a_high -d 10000 -t /mnt/nas_downloads/data/hassio/tstreamer -l DEBUG -f True -k True -u 60


#./deploy.sh -r rtsp://192.168.10.1:7447/JdCDi8nwunLfXqbD -n uvc_g3_flex_b -d 10 -t /mnt/nas_downloads/data/hassio/tstreamer -l INFO -f True -k True
#./deploy.sh -r rtsp://192.168.10.1:7447/zEVDUcica28HNKVu -n uvc_g3_instant_1
