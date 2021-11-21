#!/usr/bin/python3

import argparse
import pandas as pd
import sys
from PIL import Image
import requests
from io import BytesIO
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse


class S(BaseHTTPRequestHandler):
    def refresh_labels(self):
        print(f"Loading labels")
        labels = "/mnt/nas_downloads/data/hassio/tstreamer/labels.csv"
        self.df = pd.read_csv(labels)
        self.df.columns = ['stamp', 'uuid', 'puuid', 'entity', 'model', 'confidence', 'similarity', 'label', 'area', 'x1', 'y1', 'x2', 'y2']
        self.df['gpuuid'] = self.df['puuid'].map(self.df.set_index('uuid')['puuid'])
        print(f"Labels loaded")

    def __init__(self, http_svc, *args, **kwargs):
        self.refresh_labels()
        BaseHTTPRequestHandler.__init__(self, http_svc, *args, **kwargs)

    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "image/jpeg")
        self.end_headers()

    def _image(self, id):
        wmw=368
        wmh=368
        ratio = wmw/wmh
        uuid = str(id)
        pad = 1.5
        row = self.df[(self.df.uuid == uuid)]
        prow = self.df[(self.df.uuid == row['puuid'].values[0])]
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
        response = requests.get(url)

        return response.content

    def do_GET(self):
        if "uid=" in self.path:
            parsed = urlparse(self.path)
            id = parsed.query.split("=")[1]
            self._set_headers()
            self.wfile.write(self._image(id))
        else:
            self.send_response(404,message='Not Found')
            self.end_headers()

    def do_HEAD(self):
        self._set_headers()

def run(server_class=HTTPServer, handler_class=S, addr="192.168.10.23", port=35555):
    server_address = (addr, port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting httpd server on {addr}:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run()