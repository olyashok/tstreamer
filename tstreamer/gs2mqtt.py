#!/usr/bin/python3
from __future__ import annotations
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image, ImageDraw, ImageFont
import argparse
import io
import cv2
from shutil import copyfile
import numpy as np
import time
from datetime import datetime, tzinfo, timedelta
import json
import uuid
import sys
import time
import logging
from logging.handlers import *
import requests
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import paho.mqtt.client as mqttClient
import pytesseract
import os.path
import os

class Zone(tzinfo):
    def __init__(self,offset,isdst,name):
        self.offset = offset
        self.isdst = isdst
        self.name = name
    def utcoffset(self, dt):
        return timedelta(hours=self.offset) + self.dst(dt)
    def dst(self, dt):
            return timedelta(hours=1) if self.isdst else timedelta(0)
    def tzname(self,dt):
         return self.name

GMT = Zone(0,False,'GMT')
EST = Zone(-5,False,'EST')

class Video():
    def __init__(self, stream):
        Gst.init(None)
        self._frame = None

        self.video_source = stream
        self.video_codec = '! rtph264depay ! h264parse ! nvh264dec'
        #self.video_decode = '! decodebin ! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert'
        self.video_decode = '! video/x-raw ! videoconvert'
        self.video_sink_conf = '! appsink emit-signals=true sync=false max-buffers=2 drop=true'

        self.video_pipe = None
        self.video_sink = None

        self.run()

    def start_gst(self, config=None):
        if not config:
            self.config = \
                [
                    'videotestsrc ! decodebin',
                    '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                    '! appsink'
                ]

        command = ' '.join(config)
        self.video_pipe = Gst.parse_launch(command)
        self.video_pipe.set_state(Gst.State.PLAYING)
        self.video_sink = self.video_pipe.get_by_name('appsink0')

    @staticmethod
    def gst_to_opencv(sample):
        buf = sample.get_buffer()
        caps = sample.get_caps()
        format = caps.get_structure(0).get_value('format')
        h = caps.get_structure(0).get_value('height')
        w = caps.get_structure(0).get_value('width')
        byteArray = buf.extract_dup(0, buf.get_size())

        if format == "NV12":
            e = w * h
            Y = np.frombuffer(byteArray[0:e], dtype=np.uint8)
            Y = np.reshape(Y, (h, w))

            s = e
            V = np.frombuffer(byteArray[s::2], dtype=np.uint8)
            V = np.repeat(V, 2, 0)
            V = np.reshape(V, (int(h / 2), w))
            V = np.repeat(V, 2, 0)

            U = np.frombuffer(byteArray[s + 1::2], dtype=np.uint8)
            U = np.repeat(U, 2, 0)
            U = np.reshape(U, (int(h / 2), w))
            U = np.repeat(U, 2, 0)

            RGBMatrix = (np.dstack([Y, U, V])).astype(np.uint8)
            RGBMatrix = cv2.cvtColor(RGBMatrix, cv2.COLOR_YUV2BGR, 3)
        return RGBMatrix

    def frame(self):
        return self._frame

    def frame_available(self):
        return self._frame is not None

    def run(self):
        self.start_gst(
            [
                self.video_source,
                self.video_codec,
                self.video_decode,
                self.video_sink_conf
            ])

        self.video_sink.connect('new-sample', self.callback)

    def callback(self, sink):
        sample = sink.emit('pull-sample')
        new_frame = self.gst_to_opencv(sample)
        self._frame = new_frame

        return Gst.FlowReturn.OK


DATA_TORCHSERVE = "data_torchserve"
DATA_KEY_INPUT = ">"
DATA_KEY_ALL = "*"
DATA_KEY_OBJ = "obj"
DATA_MODEL = "model"
DATA_BOX = "bounding_box"
DATA_BOX_AREA = "box_area"
DATA_CENTROID = "centroid"
DATA_PREDICTION_TYPE = "prediction_type"
DATA_SIMILARITY_TO_LAST = "similarity"
DATA_PREDICTION_TYPE_OBJECT = "object"
DATA_PREDICTION_TYPE_CLASS = "class"
DATA_HEIGHT = "height"
DATA_WIDTH = "width"
DATA_X = "x"
DATA_XMIN = "x_min"
DATA_XMAX = "x_max"
DATA_Y = "y"
DATA_YMIN = "y_min"
DATA_YMAX = "y_max"
DATA_SCORE = "score"
DATA_NAME = "name"
DATA_CONFIDENCE = "confidence"
DATA_UNIQUE_ID = "uuid"
DATA_CROPID = "cropid"
DATA_IMAGE = "image"
DATA_PARENT_ID = "parent_id"
DATA_ENTITY_ID = "entity_id"
DATA_GRAND_PARENT_ID = "grand_parent_id"
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S"
DATA_FILE_PATH = "path"
DATA_FILE_PATH_FRAME = "path_frame"
DATA_FILE_PATH_CROP = "path_crop"

def get_objects(cropid: str, predictions: list, model: str, img_width: int, img_height: int):
    """Return objects with formatting and extra info."""
    objects = []
    decimal_places = 3

    if isinstance(predictions, dict):
        for key in predictions:
            confidence = predictions[key]
            box = {
                DATA_HEIGHT: 1,
                DATA_WIDTH: 1,
                DATA_YMIN: 0,
                DATA_XMIN: 0,
                DATA_YMAX: 1,
                DATA_XMAX: 1,
            }
            box_area = round(box[DATA_HEIGHT] * box[DATA_WIDTH], decimal_places)
            centroid = {
                DATA_X: round(box[DATA_XMIN] + (box[DATA_WIDTH] / 2), decimal_places),
                DATA_Y: round(box[DATA_YMIN] + (box[DATA_HEIGHT] / 2), decimal_places),
            }
            objects.append(
                {
                    DATA_BOX: box,
                    DATA_BOX_AREA: box_area,
                    DATA_CENTROID: centroid,
                    DATA_NAME: key,
                    DATA_CONFIDENCE: confidence * 100,
                    DATA_MODEL: model,
                    DATA_PREDICTION_TYPE: DATA_PREDICTION_TYPE_CLASS,
                    DATA_UNIQUE_ID: uuid.uuid4().hex,
                    DATA_PARENT_ID: cropid,
                    DATA_SIMILARITY_TO_LAST: -1
                }
            )
    else:
        for pred in predictions:
            if isinstance(pred, str):  # this is image class not object detection so no objects
                return objects
            name = list(pred.keys())[0]

            box_width = pred[name][2] - pred[name][0]
            box_height = pred[name][3] - pred[name][1]
            box = {
                DATA_HEIGHT: round(box_height / img_height, decimal_places),
                DATA_WIDTH: round(box_width / img_width, decimal_places),
                DATA_YMIN: round(pred[name][1] / img_height, decimal_places),
                DATA_XMIN: round(pred[name][0] / img_width, decimal_places),
                DATA_YMAX: round(pred[name][3] / img_height, decimal_places),
                DATA_XMAX: round(pred[name][2] / img_width, decimal_places),
            }
            box_area = round(box[DATA_HEIGHT] * box[DATA_WIDTH], decimal_places)
            centroid = {
                DATA_X: round(box[DATA_XMIN] + (box[DATA_WIDTH] / 2), decimal_places),
                DATA_Y: round(box[DATA_YMIN] + (box[DATA_HEIGHT] / 2), decimal_places),
            }
            confidence = round(pred[DATA_SCORE] * 100, decimal_places)

            objects.append(
                {
                    DATA_BOX: box,
                    DATA_BOX_AREA: box_area,
                    DATA_CENTROID: centroid,
                    DATA_NAME: name,
                    DATA_MODEL: model,
                    DATA_CONFIDENCE: confidence,
                    DATA_PREDICTION_TYPE: DATA_PREDICTION_TYPE_OBJECT,
                    DATA_UNIQUE_ID: uuid.uuid4().hex,
                    DATA_PARENT_ID: cropid,
                    DATA_SIMILARITY_TO_LAST: -1
                }
            )

    return sorted(objects, key=lambda i: i['bounding_box']['x_min'])


def infer_via_rest(host, port, model_name, model_input):
    """Run inference via REST."""
    url = f"http://{host}:{port}/predictions/{model_name}"
    prediction = requests.post(url, data=model_input, timeout=(5, 1)).text
    return prediction


def draw_box(
    draw: ImageDraw,
    box: tuple[float, float, float, float],
    img_width: int,
    img_height: int,
    text: str = "",
    color: tuple[int, int, int] = (255, 255, 0),
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


def process_image(client, image, args):
    """Process an image."""
    if args.test_image is not None:
        pil_image = Image.open(args.test_image)
    elif "JpegImageFile" in f"{type(image)}":
        pil_image = image
    else:
        pil_image = Image.fromarray(image)

    _image_width, _image_height = pil_image.size
    frameuuid = uuid.uuid4().hex
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG', quality=99)
    img_byte_arr = img_byte_arr.getvalue()
    images = {DATA_KEY_INPUT: [{DATA_IMAGE: img_byte_arr, DATA_CROPID: frameuuid, DATA_WIDTH: _image_width, DATA_HEIGHT: _image_height}]}

    _objects = []  # The parsed raw data
    _targets_found = []

    _models = ['> | yolov5x-1280 | {"person": ">", "boat": ">", "car,truck,bus": "car", "dog,cat,bear,teddy bear,sheep,cow": "animal", "*": "null"} | ' +
               '(\"pro_c\" not in args.name or (.2 <= obj[\"centroid\"][\"x\"] <= .78 and .23 <= obj[\"centroid\"][\"y\"] <= 1 and not (.53 <= obj[\"centroid\"][\"x\"] <= .59 and .23 <= obj[\"centroid\"][\"y\"] <= .33))) and '+ #dock area only and not boat lift
               '(\"946\" not in args.name or not (.384 <= obj[\"centroid\"][\"x\"] <= .394 and 0.185 <= obj[\"centroid\"][\"y\"] <= .195)) and '+ #exclude tree
               '(\"946\" not in args.name or not (.641 <= obj[\"centroid\"][\"x\"] <= .671 and 0.134 <= obj[\"centroid\"][\"y\"] <= .145)) and '+ #exclude weird branch
               '(\"pro_a\" not in args.name or not (.92 <= obj[\"centroid\"][\"y\"])) and '+ #exclude rock
               '(\"pro_a\" not in args.name or not (.48 <= obj[\"centroid\"][\"x\"] <= 1 and 0 <= obj[\"centroid\"][\"y\"] <= .47)) and '+ #exclude circle in driveway
               '(\"pro_a\" not in args.name or not (.230 <= obj[\"centroid\"][\"x\"] <= .240 and 0.281 <= obj[\"centroid\"][\"y\"] <= .291)) and '+ #exclude tree
               '(\"pro_a\" not in args.name or not (.47 <= obj[\"centroid\"][\"x\"] <= .63 and 0.79 <= obj[\"centroid\"][\"y\"] <= .92)) and '+ #exclude rock
               '(\"pro_b\" not in args.name or \"boat\" not in obj[\"name\"]) and '+ # exclude boats
               '(\"doorbell\" not in args.name or obj[\"box_area\"]>0.01) and '+
               '(\"doorbell\" not in args.name or not (.71 <= obj[\"centroid\"][\"x\"] <= .81 and 0.29 <= obj[\"centroid\"][\"y\"] <= .73)) and '+ #exclude girland
               '(\"g4_bullet\" not in args.name or not (.36 <= obj[\"centroid\"][\"x\"] <= .4 and 0.22 <= obj[\"centroid\"][\"y\"] <= .34)) and '+ #exclude chair
               '(obj[\"box_area\"]>0.001)', # no small objects
               'person,car,animal | unifiprotect | {"object,package": "null", "*": ">"} | *']

    if (args.read_time):
        time_crop = pil_image.crop((0, 0, 0.22 * _image_width, 0.037 * _image_height))
        #time_crop.save(f"{args.directory}/t.jpeg", format='JPEG', quality=99)
        frame_timestamp = pytesseract.image_to_string(time_crop)
        if (len(frame_timestamp)<10):
            frame_timestamp = "UNKNOWN"
        else:
            frame_timestamp = frame_timestamp[0:19].replace("\n", " ").replace("\r", " ").strip()
    else:
        frame_timestamp = "STAMP_DISABLED"

    for pipe in _models:
        pipeline = pipe.split("|")
        pipe_in = pipeline[0]
        filter_in = pipeline[3].strip()
        model = pipeline[1].strip()
        label_map_hash = {}
        if len(pipeline) > 3:
            label_map = json.loads(pipeline[2])
            for labels in label_map.keys():
                out = label_map[labels]
                ind_lables = labels.split(",")
                for label in ind_lables:
                    label_map_hash[label.strip()] = out

        labels = pipe_in.split(",")
        for label in labels:
            label = label.strip()
            if label not in images:
                continue

            current_images = images[label]
            targets_found = []
            filtered = []

            for index in range(len(current_images)):
                current_image = current_images[index][DATA_IMAGE]
                current_width = current_images[index][DATA_WIDTH]
                current_height = current_images[index][DATA_HEIGHT]
                cropid = current_images[index][DATA_CROPID]

                tic = time.perf_counter()

                response = eval(infer_via_rest(args.torchserve_ip, args.torchserve_port, model, current_image))
                toc = time.perf_counter()

                if isinstance(response, str) or (isinstance(response, dict) and 'code' in response.keys()):
                    logger.error(f"Torchsever unexpected response {response}")
                    continue

                all_objects = get_objects(cropid, response, model, current_width, current_height)

                #top1
                if len(all_objects) > 0 and all_objects[0][DATA_PREDICTION_TYPE] == DATA_PREDICTION_TYPE_CLASS:
                    all_objects = all_objects[:1]

                filtered = []
                if len(label_map_hash) > 0:
                    for obj in all_objects:
                        if obj[DATA_NAME] in label_map_hash:
                            if "null" not in label_map_hash[obj[DATA_NAME]]:
                                if DATA_KEY_INPUT not in label_map_hash[obj[DATA_NAME]]:
                                    obj[f"{DATA_NAME}_original"] = obj[DATA_NAME]
                                    obj[DATA_NAME] = label_map_hash[obj[DATA_NAME]]
                                filtered.append(obj)
                        elif "*" in label_map_hash:
                            if "null" not in label_map_hash["*"]:
                                if DATA_KEY_INPUT not in label_map_hash["*"]:
                                    obj[f"{DATA_NAME}_original"] = obj[DATA_NAME]
                                    obj[DATA_NAME] = label_map_hash["*"]
                                filtered.append(obj)
                else:
                    filtered = all_objects

                #apply filters
                all_objects = filtered
                filtered = []
                if "*" != filter_in:
                    for obj in all_objects:
                        label_filter_eval = False
                        try:
                            label_filter_eval = eval(filter_in)
                        except (RuntimeError, NameError):
                            pass
                        except (TypeError):
                            logger.error(f"Error evaluating {filter_in} against {obj}")
                            label_filter_eval = False
                        if label_filter_eval:
                            filtered.append(obj)
                else:
                    filtered = all_objects

                targets_found = filtered
                #filter for min confidence (but do not filter classifier)
                if len(targets_found) > 0 and targets_found[0][DATA_PREDICTION_TYPE] != DATA_PREDICTION_TYPE_CLASS:
                    targets_found = [obj for obj in targets_found if (obj["confidence"] > args.min_confidence)]

                if len(response) > 0:
                    logger.debug(f"Torchserve {model} on {args.name} on frame {frame_timestamp} ran in {toc-tic}s and returned {response} and was pipelined to {targets_found}")

                #pipe crops
                if len(targets_found) > 0 and targets_found[0][DATA_PREDICTION_TYPE] != DATA_PREDICTION_TYPE_CLASS:
                    for obj in targets_found:
                        if DATA_BOX in obj:
                            box = obj[DATA_BOX]
                            imc = pil_image.crop((box[DATA_XMIN] * _image_width, box[DATA_YMIN] * _image_height, box[DATA_XMAX] * _image_width, box[DATA_YMAX] * _image_height))
                            crop_width, crop_height = imc.size
                            img_byte_arr = io.BytesIO()
                            imc.save(img_byte_arr, format='JPEG')
                            img_byte_arr = img_byte_arr.getvalue()
                            if obj[DATA_NAME] not in images:
                                crops = []
                                images[obj[DATA_NAME]] = crops
                            else:
                                crops = images[obj[DATA_NAME]]
                            crops.append({DATA_IMAGE: img_byte_arr, DATA_CROPID: obj[DATA_UNIQUE_ID], DATA_WIDTH: crop_width, DATA_HEIGHT: crop_height, DATA_KEY_OBJ: obj})

                _objects.extend(filtered)
                _targets_found.extend(targets_found)

    detection_time = datetime.now(EST).strftime(DATETIME_FORMAT)

    if len(_targets_found) > 0:
        logger.debug(f"{len(_targets_found)} targets found on frame {frameuuid} with frame timestamp {frame_timestamp} ")

    if (args.save_timestamped or args.save_latest):
        save_image(args, pil_image, _targets_found, detection_time, frameuuid)
    #logger.debug(f"Post image saves")
    if len(_targets_found) > 0 and args.fire_events:
        fire_events(client, args, pil_image, _targets_found, detection_time, frameuuid)


def fire_events(client, args, img, objects, stamp, frameuuid):
    logger.debug(f"Firing events")
    obj_by_uuid = {}
    for obj in objects:
        obj_by_uuid[obj[DATA_UNIQUE_ID]] = obj

    for obj in objects:
        if obj[DATA_PARENT_ID] in obj_by_uuid:
            obj[DATA_GRAND_PARENT_ID] = obj_by_uuid[obj[DATA_PARENT_ID]][DATA_PARENT_ID]

    for obj in objects:
        if (args.detect_dups > 0 and obj[DATA_SIMILARITY_TO_LAST] < (args.detect_dups/100)) or obj[DATA_SIMILARITY_TO_LAST]<0:
            prefix = f"{args.name.lower()}"
            directory = args.directory
            crop_save_path = f"{directory}/{prefix}_{stamp}_{obj[DATA_MODEL]}_{obj[DATA_PREDICTION_TYPE]}_{obj[DATA_NAME]}_{obj[DATA_UNIQUE_ID]}.jpg"
            timestamp_save_path = f"{directory}/{prefix}_{stamp}_nobox_{frameuuid}.jpg"
            obj[DATA_FILE_PATH] = crop_save_path
            obj[DATA_ENTITY_ID] = args.name
            obj[DATA_FILE_PATH_FRAME] = timestamp_save_path

            message = json.dumps(obj)
            logger.debug(f"Sending to topic {args.mqtt_topic} message {message}")
            client.publish(args.mqtt_topic, message, 1)
        else:
            logger.debug(f"Skipping event for duplicate similarity of {obj[DATA_SIMILARITY_TO_LAST]}")


def save_image(args, img, objects, stamp, frameuuid):
    """Save image files."""
    """Draws the actual bounding box of the detected objects."""
    imgc = img.copy()

    if args.show_boxes:
        draw = ImageDraw.Draw(img)

    saved_crops = {}
    saved_crops_pil = {}
    saved_crops_pad = {}
    saved_crops_pil_pad = {}
    prefix = f"{args.name.lower()}"
    directory = args.directory

    obj_by_puuid = {}
    for obj in objects:
        obj_by_puuid[obj[DATA_PARENT_ID]] = obj

    savebox = False

    counter = {}

    box_save_path_latest = (f"{directory}/{prefix}_latest_box.jpg")
    all_box_save_path_latest = (f"{directory}/all_latest_box.jpg")
    nobox_save_path_latest = (f"{directory}/{prefix}_latest_nobox.jpg")
    all_nobox_save_path_latest = (f"{directory}/all_latest_nobox.jpg")
    all_latest_crop = (f"{directory}/all_latest_crop.jpg")

    has_boxes = False

    for obj in objects:
        inc = 0
        label = obj[DATA_NAME]
        if label in counter:
            inc = counter[label] + 1
        counter[label] = inc
        confidence = obj[DATA_CONFIDENCE]
        model = obj[DATA_MODEL]
        prediction_type = obj[DATA_PREDICTION_TYPE]
        box = obj[DATA_BOX]
        box_area = obj[DATA_BOX_AREA]
        centroid = obj[DATA_CENTROID]
        predid = obj[DATA_UNIQUE_ID]
        imageid = obj[DATA_PARENT_ID]
        entity_id = prefix

        crop_save_path = f"{directory}/{prefix}_{stamp}_{model}_{prediction_type}_{label}_{predid}.jpg"
        crop_save_path_pad = f"{directory}/{prefix}_{stamp}_{model}_{prediction_type}_{label}_{predid}_pad.jpg"
        crop_save_path_latest = f"{directory}/{prefix}_latest_{prediction_type}_{label}_{inc}.jpg"
        crop_save_path_latest_pad = f"{directory}/{prefix}_latest_{prediction_type}_crop.jpg"
        all_crop_save_path_latest_pad = f"{directory}/all_latest_{prediction_type}_crop.jpg"
        all_crop_save_path_latest = f"{directory}/all_latest_{prediction_type}_{label}_{inc}.jpg"
        allni_crop_save_path_latest = f"{directory}/all_latest_{prediction_type}_{label}.jpg"
        box_save_path = f"{directory}/{prefix}_{stamp}_box_{frameuuid}.jpg"
        nobox_save_path = f"{directory}/{prefix}_{stamp}_nobox_{frameuuid}.jpg"

        if args.show_boxes:
            logger.debug(f"Drawing boxes")
            has_boxes = True
            if prediction_type == DATA_PREDICTION_TYPE_CLASS:
                box_colour = (255, 255, 0)
            else:
                box_colour = (0, 255, 0)

            box_label = f"{model}.{label}: {confidence:.1f}%"
            if predid in obj_by_puuid:
                box_label = f"{obj_by_puuid[predid][DATA_MODEL]}.{obj_by_puuid[predid][DATA_NAME]}: {obj_by_puuid[predid][DATA_CONFIDENCE]:.1f}%"
                box_colour = (0, 255, 0)

            draw_box(
                draw,
                (box[DATA_YMIN], box[DATA_XMIN], box[DATA_YMAX], box[DATA_XMAX]),
                img.width,
                img.height,
                text=box_label,
                color=box_colour,
            )

            # draw bullseye
            draw.text(
                (centroid[DATA_X] * img.width, centroid[DATA_Y] * img.height),
                text="X",
                fill=box_colour,
            )

        if args.save_crops:
            if prediction_type == DATA_PREDICTION_TYPE_OBJECT:
                imc = imgc.crop((box[DATA_XMIN] * img.width, box[DATA_YMIN] * img.height, box[DATA_XMAX] * img.width, box[DATA_YMAX] * img.height))

                wmw=368
                wmh=368
                ratio = wmw/wmh
                pad = 1.5
                w, h = img.width, img.height
                x1, y1, x2, y2 = w*box[DATA_XMIN], h*box[DATA_YMIN], w*box[DATA_XMAX], h*box[DATA_YMAX]
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
                imcp = img.crop((x1, y1, x2, y2))
                #watermark
                draw = ImageDraw.Draw(imcp)
                fz = int(100 * imcp.width / 600)
                font = ImageFont.truetype("/mnt/nas_downloads/deepstack/tstreamer/tstreamer/arial.ttf", fz)
                #draw.text((0, 0), datetime.now(EST).strftime("%H:%M"), font=font, fill="#39ff14", stroke_width=2, stroke_fill="#39ff14")
                draw.text((0, 0), datetime.now(EST).strftime("%H:%M"), font=font, fill="#f93822")

                saved_crops_pil[predid] = imc
                saved_crops_pil_pad[predid] = imcp
                obj[DATA_FILE_PATH] = f"{crop_save_path}"
                obj[DATA_FILE_PATH_CROP] = f"{crop_save_path_pad}"
                saved_crops[predid] = obj[DATA_FILE_PATH]
                saved_crops_pad[predid] = obj[DATA_FILE_PATH_CROP]
            else:
                imc = saved_crops_pil[imageid]
                imcp = saved_crops_pil_pad[imageid]
                obj[DATA_FILE_PATH] = saved_crops[imageid]
                obj[DATA_FILE_PATH_CROP] = saved_crops_pad[imageid]

        if args.detect_dups > 0:
            if os.path.exists(crop_save_path_latest):
                score = -2
                try:
                    current_im = cv2.cvtColor(np.asarray(imc),cv2.COLOR_RGB2BGR)
                    #grayA = cv2.cvtColor(np.float32(imc), cv2.COLOR_RGB2GRAY)
                    last_im = cv2.imread(crop_save_path_latest)
                    if current_im.shape[0] != last_im.shape[0] or current_im.shape[1] != last_im.shape[1]:
                        last_im = cv2.resize(last_im, (current_im.shape[1], current_im.shape[0]), interpolation = cv2.INTER_AREA)

                    #grayA = cv2.cvtColor(current_im, cv2.COLOR_RGB2GRAY)
                    #grayB = cv2.cvtColor(last_im, cv2.COLOR_BGR2GRAY)
                    #(score, diff) = compare_ssim(grayA, grayB, full=True)

                    #histogramA = cv2.calcHist([current_im], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    #histogramB = cv2.calcHist([last_im], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    #score = cv2.compareHist(histogramA, histogramB, cv2.HISTCMP_CORREL)

                    surf = cv2.ORB_create()
                    kpA, desc_a = surf.detectAndCompute(current_im, None)
                    kpB, desc_b = surf.detectAndCompute(last_im, None)
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(desc_a, desc_b)
                    similar_regions = [i for i in matches if i.distance < 70]
                    if len(matches) != 0:
                        score = len(similar_regions) / len(matches)
                except cv2.error as err:
                    logger.warning(f"Unexpected CV2 error. Skipping similarity check. Err={err}")

                obj[DATA_SIMILARITY_TO_LAST] = score

                if score*100 >= args.detect_dups:
                    saved_crops[predid] = crop_save_path_latest
                    continue

                logger.debug(f"New image identified with score {score} in {crop_save_path}")

        savebox = True

        if args.save_labels:
            logger.debug(f"Saving labels")
            label_path = f"{directory}/labels.csv"
            with open(label_path, "a+") as f:
                f.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    stamp,
                    predid,
                    imageid,
                    entity_id,
                    model,
                    confidence,
                    obj[DATA_SIMILARITY_TO_LAST],
                    label,
                    box_area,
                    int(box[DATA_XMIN] * img.width), int(box[DATA_YMIN] * img.height), int(box[DATA_XMAX] * img.width), int(box[DATA_YMAX] * img.height),
                ))


        if args.save_crops:
            logger.debug(f"Saving crops")
            if prediction_type == DATA_PREDICTION_TYPE_OBJECT:
                if args.save_timestamped:
                    imc.save(crop_save_path)
                    imcp.save(crop_save_path_pad)
                if args.save_latest:
                    imc.save(crop_save_path_latest)
                    imc.save(all_crop_save_path_latest)
                    imc.save(allni_crop_save_path_latest)
                    imcp.save(crop_save_path_latest_pad)
                    imcp.save(all_crop_save_path_latest_pad)
                    imcp.save(all_latest_crop)
                #_LOGGER.debug("Torchserve saved crops")
            else:
                obj[DATA_FILE_PATH] = saved_crops[imageid]
                if args.save_timestamped:
                    copyfile(obj[DATA_FILE_PATH], crop_save_path)
                    copyfile(obj[DATA_FILE_PATH_CROP], crop_save_path_pad)
                if args.save_latest:
                    copyfile(obj[DATA_FILE_PATH], crop_save_path_latest)
                    copyfile(obj[DATA_FILE_PATH_CROP], crop_save_path_latest)
                    copyfile(obj[DATA_FILE_PATH_CROP], all_crop_save_path_latest_pad)
                    copyfile(obj[DATA_FILE_PATH], all_crop_save_path_latest)
                    copyfile(obj[DATA_FILE_PATH], allni_crop_save_path_latest)



    if args.save_latest:
        if args.show_boxes and has_boxes:
            img.save(box_save_path_latest)
            img.save(all_box_save_path_latest)
        imgc.save(nobox_save_path_latest)
        imgc.save(all_nobox_save_path_latest)

    if savebox:
        if args.show_boxes and has_boxes:
            logger.debug(f"Saving crops")
            img.save(box_save_path)
        imgc.save(nobox_save_path)


mqtt_connected = False


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        #logger.info("Connected to broker")
        global mqtt_connected
        mqtt_connected = True
    else:
        logger.critical("Connection failed")


if __name__ == '__main__':

    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'

    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', required=True, type=str, metavar='RTSP', help='RTSP stream URL')
    parser.add_argument('--test-image', required=False, type=str, metavar='TI', help='Torchserve image')
    parser.add_argument('--torchserve-ip', required=True, type=str, metavar='TSP', help='Torchserve serve IP')
    parser.add_argument('--torchserve-port', required=False, default=8080, type=int, metavar='TSP', help='torch serve port (default:8080)')
    parser.add_argument('--name', required=True, type=str, metavar='SN', help='Sensor name')
    parser.add_argument('--min-confidence', required=False, default=60, type=int, metavar='C', help='Minimum confidence (default:60)')
    parser.add_argument('--directory', required=False, type=str, metavar='CD', help='Crop directory name')
    parser.add_argument('--fire-events', type=boolean_string, default=True, help='fire events (default:True)')
    parser.add_argument('--save-timestamped', type=boolean_string, default=True, help='save timestamped (default:False)')
    parser.add_argument('--save-crops', type=boolean_string, default=False, help='save crops (default:False)')
    parser.add_argument('--save-latest', type=boolean_string, default=False, help='save latest version (default:False)')
    parser.add_argument('--detect-dups', type=int, default='0', metavar='U', help='0-100 dups threshold')
    parser.add_argument('--save-labels', type=boolean_string, default=False, help='save labels (default:False)')
    parser.add_argument('--save-frame', type=boolean_string, default=False, help='save last frame (default:False)')
    parser.add_argument('--show-boxes', type=boolean_string, default=False, help='show boxes (default:False)')
    parser.add_argument('--read-time', type=boolean_string, default=True, help='read time (default:False)')
    parser.add_argument('--mqtt-ip', required=True, type=str, metavar='MQTTPIP', help='MQTT IP')
    parser.add_argument('--mqtt-port', required=False, default=1883, type=int, metavar='MQTTPORT', help='MQTT port (default:1883)')
    parser.add_argument('--mqtt-user', required=False, type=str, metavar='MQTTUSR', help='MQTT user')
    parser.add_argument('--mqtt-password', required=False, type=str, metavar='MQTTPWD', help='MQTT password')
    parser.add_argument('--mqtt-topic', required=False, type=str, default='gs2mqtt', metavar='MQTTTOPIC', help='MQTT Topics')
    parser.add_argument('--debug', required=False, type=str, default='CRITICAL', metavar='L', help='Logging level')
    parser.add_argument('--delay', required=False, type=int, default='1000', metavar='D', help='Delay in ms')
    args = parser.parse_args()

    logger = logging.getLogger(f"gs2mqtt.{args.name}")
    logger.setLevel(args.debug)
    fh = RotatingFileHandler(f'{args.directory}/gs2mqtt.{args.name}.log', maxBytes=(1048576*10), backupCount=7)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Starting streamer")

    client = mqttClient.Client("Python")
    client.username_pw_set(args.mqtt_user, password=args.mqtt_password)
    client.on_connect = on_connect
    client.connect(args.mqtt_ip, port=args.mqtt_port)
    client.loop_start()

    logger.debug("MQTT loop started")

    while not mqtt_connected:
        time.sleep(0.1)

    if "rtsp" in args.stream:
        video = Video(args.stream)
        logger.info("Video stream created")

    i = 1
    loop = True
    while loop:
        # Wait for the next frame
        if "rtsp" in args.stream:
            if not video.frame_available():
                continue

        if "rtsp" in args.stream:
            frame = video.frame()
        elif "http" in args.stream:
            frame = Image.open(requests.get(args.stream, stream=True).raw)
        else:
            frame = Image.open(args.stream)
            loop = False

        tic = time.perf_counter()
        try:
            process_image(client, frame, args)
        except OSError as err:
            logger.debug(f"OS error {err}")
            raise
        except:
            logger.debug(f"Unexpected error {sys.exc_info()}")
            raise

        if args.save_frame:
            frame_save_path = f"{args.directory}/{args.name}_last_frame.jpg"
            if "rtsp" in args.stream:
                pil_image = Image.fromarray(frame)
                pil_image.save(frame_save_path)
            else:
                frame.save(frame_save_path)

        toc = time.perf_counter()

        #logger.debug(f"Processed frame {i} on {args.name} in {toc-tic}s")

        i = i + 1

        time.sleep(args.delay/1000)

        #if i > 1:
        #    quit()
