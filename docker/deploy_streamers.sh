#!/bin/sh

#approach
cd /mnt/nas_downloads/deepstack/tstreamer2/docker/

./deploy.sh http://192.168.10.184/snap.jpeg uvc_g4_pro_946e_high 5000
#doorbell
./deploy.sh http://192.168.10.217/snap.jpeg uvc_g4_doorbell_high 4000
#driveway
./deploy.sh http://192.168.10.246/snap.jpeg uvc_g3_pro_a_high 2000
#dock
./deploy.sh http://192.168.10.198/snap.jpeg uvc_g3_pro_c_high 15000
#pool
./deploy.sh http://192.168.10.214/snap.jpeg uvc_g3_pro_b_high 10000
#garage
./deploy.sh http://192.168.10.193/snap.jpeg uvc_g3_flex_a_high 10000
#deck
./deploy.sh http://192.168.10.191/snap.jpeg uvc_g4_bullet_a_high 10000
# poolhouse
./deploy.sh http://192.168.10.203/snap.jpeg uvc_g4_bullet_b_high 10000
#nursery
#./deploy.sh http://192.168.10.159/snap.jpeg uvc_g3_flex_b_high 20000


#./deploy.sh rtsp://192.168.10.1:7447/JdCDi8nwunLfXqbD uvc_g3_flex_b 10
#./deploy.sh rtsp://192.168.10.1:7447/zEVDUcica28HNKVu uvc_g3_instant_1
