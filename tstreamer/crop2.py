#!/usr/bin/python3

import argparse
import pandas as pd
import sys
from PIL import Image
import requests
from io import BytesIO
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

print(f"Loading labels")
labels = "/mnt/nas_downloads/data/hassio/tstreamer/labels.csv"
df = pd.read_csv(labels)
df.columns = ['stamp', 'uuid', 'puuid', 'entity', 'model', 'confidence', 'similarity', 'label', 'area', 'x1', 'y1', 'x2', 'y2']
df['gpuuid'] = df['puuid'].map(df.set_index('uuid')['puuid'])
print(f"Labels loaded")

id = "4f913748bac74291acb957e12cfad60e"

wmw=368
wmh=368
ratio = wmw/wmh
uuid = str(id)
pad = 1.5
row = df[(df.uuid == uuid)]
prow = df[(df.uuid == row['puuid'].values[0])]
filename = f"{row['entity'].values[0]}_{row['stamp'].values[0].replace(' ', '-')}_box_{row['gpuuid'].values[0]}.jpg"
w, h = row['x2'].values[0], row['y2'].values[0]
x1, y1, x2, y2 = prow['x1'].values[0], prow['y1'].values[0], prow['x2'].values[0], prow['y2'].values[0]
cw = x2 - x1
ch = y2 - y1
ccx = x1 + cw/2
ccy = y1 + ch/2
ecw = cw*pad
ech = ch*pad
if ecw>=ech:
    ech = ecw*ratio
else:
    ecw = ech*ratio
eccx = ccx
eccy = ccy
if (eccx<ecw/2): eccx = ecw/2
if (eccx>(w-ecw/2)): eccx = w-ecw/2
if (eccy<ech/2): eccy = ech/2
if (eccy>(h-ech/2)): eccy = h-ech/2
x1 = max(0,eccx-ecw/2)
y1 = max(0,eccy-ech/2)
x2 = min(eccx+ecw/2,w)
y2 = min(eccy+ech/2,h)
args = f"bgcolor=black&width={wmw}&height={wmh}&crop={x1},{y1},{x2},{y2}"
url=f"http://192.168.10.23:39876/ha/tstreamer/{filename}?{args}"
print(url)
