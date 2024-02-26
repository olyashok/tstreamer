#!/bin/sh

find /mnt/localshared/data/hassio/tstreamer -type f -mtime +7 -regextype posix-extended -regex '.*[a-z0-9]{32}.*' -delete
find /mnt/localshared/data/hassio/tstreamer -type f -mtime +180 -delete
