#!/usr/bin/python3
from __future__ import annotations
from PIL import Image, ImageDraw
from shutil import copyfile
import argparse
import io
import cv2
import numpy as np
import time
import datetime
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
                    DATA_PARENT_ID: cropid
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
                    DATA_PARENT_ID: cropid
                }
            )
    return objects


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

    _models = ['> | fastrcnn | {"person": ">", "car,truck,bus": "car", "dog,cat,bear,teddy bear,sheep,cow": "animal", "*": "null"} | (\"946e\" not in args.name or obj[\"centroid\"][\"y\"]>0.46) and (obj[\"confidence\"]>76) and (\"car\" not in obj[\"name\"] or obj[\"box_area\"]>0.01)',
               'person,car,animal | unifiprotect | {"object,package": "null", "*": ">"} | *']

    if (args.read_time):
        time_crop = pil_image.crop((0, 0, 0.17 * _image_width, 0.037 * _image_height))
        #time_crop.save(f"{args.directory}/t.jpeg", format='JPEG', quality=99)
        frame_timestamp = pytesseract.image_to_string(time_crop)
        if (len(frame_timestamp)<10):
            frame_timestamp = "UNKNOWN"
        else:
            frame_timestamp = frame_timestamp[0:19]
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

                logger.debug(f"Torchserve {model} on {args.name} on frame {frame_timestamp} ran in {toc-tic}s and returned {response}")

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
                targets_found = [obj for obj in targets_found if (obj["confidence"] > args.min_confidence)]

                logger.debug(f"Torchserve {model} on {args.name} on frame {frame_timestamp} ran in {toc-tic}s and was pipelined to {targets_found}")

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

    detection_time = datetime.datetime.now().strftime(DATETIME_FORMAT)

    logger.debug(f"{len(_targets_found)} targets found on frame {frameuuid} with frame timestamp {frame_timestamp} ")

    if len(_targets_found) > 0 and (args.save_timestamped or args.save_latest):
        save_image(args, pil_image, _targets_found, detection_time, frameuuid)

    if len(_targets_found) > 0 and args.fire_events:
        fire_events(client, args, pil_image, _targets_found, detection_time, frameuuid)


def fire_events(client, args, img, objects, stamp, frameuuid):
    obj_by_uuid = {}
    for obj in objects:
        obj_by_uuid[obj[DATA_UNIQUE_ID]] = obj

    for obj in objects:
        if obj[DATA_PARENT_ID] in obj_by_uuid:
            obj[DATA_GRAND_PARENT_ID] = obj_by_uuid[obj[DATA_PARENT_ID]][DATA_PARENT_ID]

    for obj in objects:
        prefix = f"{args.name.lower()}"
        directory = args.directory
        crop_save_path = f"{directory}/{prefix}_{stamp}_{obj[DATA_MODEL]}_{obj[DATA_PREDICTION_TYPE]}_{obj[DATA_NAME]}_{obj[DATA_UNIQUE_ID]}.jpg"
        timestamp_save_path = f"{directory}/{prefix}_{stamp}_nobox_{frameuuid}.jpg"
        obj[DATA_FILE_PATH] = crop_save_path
        obj[DATA_ENTITY_ID] = args.name
        obj[DATA_FILE_PATH_FRAME] = timestamp_save_path
        message = json.dumps(obj)
        logger.debug(f"{message} sent to topic {args.mqtt_topic}")
        client.publish(args.mqtt_topic, message)


def save_image(args, img, objects, stamp, frameuuid):
    """Save image files."""
    """Draws the actual bounding box of the detected objects."""
    imgc = img.copy()

    if args.show_boxes:
        draw = ImageDraw.Draw(img)

    saved_crops = {}
    prefix = f"{args.name.lower()}"
    directory = args.directory

    obj_by_puuid = {}
    for obj in objects:
        obj_by_puuid[obj[DATA_PARENT_ID]] = obj

    for obj in objects:
        label = obj[DATA_NAME]
        confidence = obj[DATA_CONFIDENCE]
        model = obj[DATA_MODEL]
        prediction_type = obj[DATA_PREDICTION_TYPE]
        box = obj[DATA_BOX]
        box_area = obj[DATA_BOX_AREA]
        centroid = obj[DATA_CENTROID]
        predid = obj[DATA_UNIQUE_ID]
        imageid = obj[DATA_PARENT_ID]
        entity_id = prefix

        if args.save_labels:
            label_path = f"{directory}/labels.csv"
            with open(label_path, "a+") as f:
                f.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    stamp,
                    predid,
                    imageid,
                    entity_id,
                    model,
                    confidence,
                    label,
                    box_area,
                    int(box[DATA_XMIN] * img.width), int(box[DATA_YMIN] * img.height), int(box[DATA_XMAX] * img.width), int(box[DATA_YMAX] * img.height),
                ))
            #_LOGGER.debug("Torchserve saved labels")

        if args.save_crops:
            if prediction_type == DATA_PREDICTION_TYPE_OBJECT:
                imc = imgc.crop((box[DATA_XMIN] * img.width, box[DATA_YMIN] * img.height, box[DATA_XMAX] * img.width, box[DATA_YMAX] * img.height))
                if args.save_timestamped:
                    crop_save_path = f"{directory}/{prefix}_{stamp}_{model}_{prediction_type}_{label}_{predid}.jpg"
                    imc.save(crop_save_path)
                    obj[DATA_FILE_PATH] = f"{crop_save_path}"
                    saved_crops[predid] = obj[DATA_FILE_PATH]
                if args.save_latest:
                    crop_save_path = f"{directory}/{prefix}_latest_{prediction_type}_{label}.jpg"
                    imc.save(crop_save_path)
                #_LOGGER.debug("Torchserve saved crops")
            else:
                obj[DATA_FILE_PATH] = saved_crops[imageid]
                classified_crop_path = f"{directory}/{prefix}_{stamp}_{model}_{prediction_type}_{label}_{predid}.jpg"
                copyfile(obj[DATA_FILE_PATH], classified_crop_path)

        if args.show_boxes and prediction_type == DATA_PREDICTION_TYPE_OBJECT:
            box_colour = (255, 255, 0)

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

    if len(objects) > 0:
        suffix = f""

        if args.save_latest:
            if args.show_boxes:
                latest_save_path = (f"{directory}/{prefix}_latest_box{suffix}.jpg")
                img.save(latest_save_path)
            latest_save_path = f"{directory}/{prefix}_latest_nobox{suffix}.jpg"
            imgc.save(latest_save_path)

        if args.save_timestamped:
            if args.show_boxes:
                timestamp_save_path = f"{directory}/{prefix}_{stamp}_box{suffix}_{frameuuid}.jpg"
                img.save(timestamp_save_path)
            timestamp_save_path = f"{directory}/{prefix}_{stamp}_nobox{suffix}_{frameuuid}.jpg"
            imgc.save(timestamp_save_path)
        #_LOGGER.debug("Torchserve saved uncropped images")


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

    video = Video(args.stream)

    logger.info("Video stream created")

    i = 1
    while True:
        # Wait for the next frame
        if not video.frame_available():
            continue

        frame = video.frame()
        tic = time.perf_counter()
        process_image(client, frame, args)

        if args.save_frame:
            frame_save_path = f"{args.directory}/{args.name}_last_frame.jpg"
            pil_image = Image.fromarray(frame)
            pil_image.save(frame_save_path)

        toc = time.perf_counter()

        logger.debug(f"Processed frame {i} on {args.name} in {toc-tic}s")

        i = i + 1

        time.sleep(args.delay/1000)

        #if i > 1:
        #    quit()
