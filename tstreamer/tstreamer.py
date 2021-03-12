import io
import itertools
import subprocess
import time
import urllib.request
from threading import Thread
from typing import Tuple
from os import path

import cv2
import grpc
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw

import streamlit as st

st.set_page_config(layout="wide")

TEST_IMAGES = {
    "-": "",
    "Front 3840x2160 cam": ["http://192.168.10.211/snap.jpeg", "rtsp://192.168.10.1:7447/ySci8n2TNfOncXMb"],
    "Front 640x360 cam": ["http://192.168.10.211/snap.jpeg", "rtsp://192.168.10.1:7447/ahoCPkXFexpDjXAS"],
    "Wifi 640x360 cam": ["http://192.168.10.190/snap.jpeg", "rtsp://192.168.10.1:7447/aVw23e6tyMAWliBD"],
    "Backyard cam": ["http://192.168.10.232/snap.jpeg", "rtsp://192.168.10.1:7447/ahoCPkXFexpDjXAS"],
    "B 1920x1028 cam ": ["http://192.168.10.197/snap.jpeg", "rtsp://192.168.10.1:7447/JdCDi8nwunLfXqbD"],
    "B 640x360 cam ": ["http://192.168.10.197/snap.jpeg", "rtsp://192.168.10.1:7447/fZSTd3E7sQ7AWt7Q"],
    "Doorbell 1600x1200 cam": ["http://192.168.10.198/snap.jpeg", "rtsp://192.168.10.1:7447/cshR3cmt8nir4yEh"],
    "Doorbell 480x360 cam": ["http://192.168.10.198/snap.jpeg", "rtsp://192.168.10.1:7447/pc0k43FyuEx66TFJ"],
}

CONTAINER = "tstreamer"

RED = (255, 0, 0)  # For objects within the ROI
GREEN = (0, 255, 0)  # For ROI box
YELLOW = (255, 255, 0)  # For objects outside the ROI


def infer(stub, model_name, model_input):
    url = 'http://localhost:8080/predictions/{}'.format(model_name)
    prediction = requests.post(url, data=model_input).text
    return prediction


def get_objects(predictions: list, img_width: int, img_height: int):
    """Return objects with formatting and extra info."""
    objects = []
    decimal_places = 3
    for pred in predictions:
        if isinstance(pred, str):  # this is image class not object detection so no objects
            return objects
        name = list(pred.keys())[0]

        box_width = pred[name][2]-pred[name][0]
        box_height = pred[name][3]-pred[name][1]
        box = {
            "height": round(box_height / img_height, decimal_places),
            "width": round(box_width / img_width, decimal_places),
            "y_min": round(pred[name][1] / img_height, decimal_places),
            "x_min": round(pred[name][0] / img_width, decimal_places),
            "y_max": round(pred[name][3] / img_height, decimal_places),
            "x_max": round(pred[name][2] / img_width, decimal_places),
        }
        box_area = round(box["height"] * box["width"], decimal_places)
        centroid = {
            "x": round(box["x_min"] + (box["width"] / 2), decimal_places),
            "y": round(box["y_min"] + (box["height"] / 2), decimal_places),
        }
        confidence = round(pred['score'], decimal_places)

        objects.append(
            {
                "bounding_box": box,
                "box_area": box_area,
                "centroid": centroid,
                "name": name,
                "confidence": confidence,
            }
        )
    return objects


def draw_box(
    draw: ImageDraw,
    box: Tuple[float, float, float, float],
    img_width: int,
    img_height: int,
    text: str = "",
    color: Tuple[int, int, int] = (255, 255, 0),
) -> None:
    """
    Draw a bounding box on and image.
    The bounding box is defined by the tuple (y_min, x_min, y_max, x_max)
    where the coordinates are floats in the range [0.0, 1.0] and
    relative to the width and height of the image.
    For example, if an image is 100 x 200 pixels (height x width) and the bounding
    box is `(0.1, 0.2, 0.5, 0.9)`, the upper-left and bottom-right coordinates of
    the bounding box will be `(40, 10)` to `(180, 50)` (in (x,y) coordinates).
    """

    line_width = 3
    font_height = 8
    y_min, x_min, y_max, x_max = box
    (left, right, top, bottom) = (
        x_min * img_width,
        x_max * img_width,
        y_min * img_height,
        y_max * img_height,
    )
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=line_width,
        fill=color,
    )
    if text:
        draw.text(
            (left + line_width, abs(top - line_width - font_height)), text, fill=color
        )


images = TEST_IMAGES
pick_img = st.sidebar.radio("Which image?", [x for x in images.keys()])

if pick_img:
    item = images[pick_img]
    if isinstance(item, str):
        filename = item
        stream = None
    else:
        filename = item[0]
        stream = item[1]


col1, col2 = st.beta_columns(2)

image_placeholder = col1.empty()
frame_placeholder = col1.empty()
timer_placeholder = col2.empty()
data_placeholder = col2.empty()

model="fastrcnn"

if stream is not None:
    video = cv2.VideoCapture(f"rtspsrc location={stream} ! decodebin ! videoconvert ! appsink max-buffers=1 drop=true")
    fps = video.get(cv2.CAP_PROP_FPS)

tic = time.perf_counter()
frameIdProcStart = 0
frameIdProc = 0

while True:
    if stream is not None:
        success, npimage = video.read()
        #npimage = cv2.cvtColor(npimage, cv2.COLOR_BGR2RGB)
        frameIdProc = frameIdProc + 1
        fps_actual = 1/(time.perf_counter()-tic) * (frameIdProc-frameIdProcStart)
        frame_placeholder.write(f"{fps_actual:0.2f} actual FPS vs stream {fps} FPS")
        frameIdProcStart = frameIdProc
        tic = time.perf_counter()
        pil_image = Image.fromarray(npimage)
    else:
        break

    if model is not None and model != "none":
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        draw = ImageDraw.Draw(pil_image)

        infers = time.perf_counter()
        response = eval(infer("", model, img_byte_arr))
        inferf = time.perf_counter()
        objects = get_objects(response, pil_image.width, pil_image.height)

        if True:
            for obj in objects:
                name = obj["name"]
                confidence = obj["confidence"]
                box = obj["bounding_box"]
                box_label = f"{name} {confidence}"
                draw_box(draw, (box["y_min"], box["x_min"], box["y_max"], box["x_max"]),
                        pil_image.width, pil_image.height, text=box_label, color=YELLOW,)

        if response is list or isinstance(response, dict):
            df = pd.DataFrame(response.items())
        elif isinstance(response, str):
            col2.write(response)
        else:
            df = pd.DataFrame()
            for pred in response:
                label = list(pred.keys())[0]
                row = [label, f"{pred['score']:0.2f}", f"{pred[label][0]:0.0f}",
                       f"{pred[label][1]:0.0f}", f"{pred[label][2]:0.0f}", f"{pred[label][3]:0.0f}"]
                row = pd.DataFrame(row).T
                df = df.append(row)
                # col2.write(row)

        timer_placeholder.write(
            f"Infer in {inferf - infers:0.4f}s or {1/(inferf-infers):0.4f} FPS")
        if df is not None:
            data_placeholder.write(df)

    image_placeholder.image(np.array(pil_image),
                            caption="Processed image", use_column_width=True,)
    if stream is None:
        break
