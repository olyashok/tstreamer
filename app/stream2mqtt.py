import io
import time
import uuid
from PIL import Image, ImageDraw, ImageFont
import pytz
import requests
import ultralytics
import paho.mqtt.client as mqttClient
from datetime import datetime, tzinfo, timedelta
import sys
import logging
import pytesseract
import os.path
import os
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import argparse
from logging.handlers import *
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import json

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S"

mqtt_connected = False
mqtt_client = None
model = None

def draw_box(
    draw: ImageDraw,
    box: tuple[float, float, float, float],
    img_width: int,
    img_height: int,
    text: str = "",
    color: tuple[int, int, int] = (255, 255, 0),
) -> None:
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

def process_image(image, args):
    global model
    pil_image = image
    frameuuid = uuid.uuid4().hex
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG', quality=99)
    img_byte_arr = img_byte_arr.getvalue()

    EXCLUDES=""
    output=""
    for line in open(args.exclude):
        li=line.strip()
        if not li.startswith("#"):
            output += li
    if (output!=EXCLUDES):
        EXCLUDES = output

    _models = ['> | yolov8x | {"person": ">", "boat": ">", "car,truck,bus": "car", "dog,cat,bear,teddy bear,sheep,cow": "animal", "*": "null"} | ' +
               EXCLUDES +
               '(\"doorbell\" not in args.name or obj[\"box_area\"]>0.01) and '+
               '(obj[\"box_area\"]>0.001)'
               ]
    _objects = []  # The parsed raw data
    _targets_found = []

    for pipe in _models:
        pipeline = pipe.split("|")
        pipe_in = pipeline[0]
        modelname = pipeline[1].strip()
        filter_in = pipeline[3].strip()
        label_map_hash = {}
        if len(pipeline) > 3:
            label_map = json.loads(pipeline[2])
            for labels in label_map.keys():
                out = label_map[labels]
                ind_lables = labels.split(",")
                for label in ind_lables:
                    label_map_hash[label.strip()] = out

        if model is None:
            model = YOLO(f"{args.triton_url}/{modelname}/versions/1/infer", task='detect')
        labels = json.load(open("/app/yolov8.labels.json"))
        labels = list(labels.values())
        results = model(image, stream=False, imgsz=1024)
        result_boxes = []
        for r in results:
            r.names = labels
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            for box in r.boxes:
                label = labels[int(box.cls)]
                confidence = float(box.conf)
                xyxy = box.xyxy.numpy()[0]
                x1, y1, x2, y2 = xyxy
                x1, y1, x2, y2 = x1 / image.width, y1 / image.height, x2 / image.width, y2 / image.height
                box_area = (x2 - x1) * (y2 - y1)
                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                result_boxes.append({"name": label,
                                     "confidence": confidence*100,
                                     "bounding_box": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2, "height": y2-y1, "width": x2-x1},
                                     "box_area": box_area,
                                     "model": modelname,
                                     "parent_id": frameuuid,
                                     "uuid": uuid.uuid4().hex,
                                     "centroid": {"x": centroid[0], "y": centroid[1]}})

        all_objects = result_boxes
        filtered = []
        if len(label_map_hash) > 0:
            for obj in all_objects:
                if obj['name'] in label_map_hash:
                    if "null" not in label_map_hash[obj['name']]:
                        if ">" not in label_map_hash[obj['name']]:
                            obj[f"{'name'}_original"] = obj['name']
                            obj['name'] = label_map_hash[obj['name']]
                        filtered.append(obj)
                elif "*" in label_map_hash:
                    if "null" not in label_map_hash["*"]:
                        if ">" not in label_map_hash["*"]:
                            obj[f"{'name'}_original"] = obj['name']
                            obj['name'] = label_map_hash["*"]
                        filtered.append(obj)
        else:
            filtered = all_objects

        filtered = sorted(filtered, key=lambda k: k['confidence'], reverse=True)
        # get 1st object
        filtered = filtered[:1]

        # if len(result_boxes) > 0:
        #     logger.debug(f"{modelname} on {args.name} returned {len(result_boxes)} detections")

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

        if len(result_boxes) > 0:
            logger.debug(f"{modelname} on {args.name} returned {len(result_boxes)} detections and was pipelined to {filtered}")

        targets_found = filtered
        #filter for min confidence (but do not filter classifier)
        if len(targets_found) > 0:
            targets_found = [obj for obj in targets_found if (obj["confidence"] > args.min_confidence)]

        if len(result_boxes) > 0:
            logger.debug(f"{modelname} on {args.name} returned {len(result_boxes)} detections and was filtered to {targets_found}")

        _objects.extend(filtered)
        _targets_found.extend(targets_found)

    detection_time = datetime.now(pytz.timezone('US/Eastern')).strftime(DATETIME_FORMAT)
    if len(_targets_found) > 0:
        logger.debug(f"{len(_targets_found)} targets found on frame {frameuuid} with frame timestamp {detection_time} ")
    save_image(args, pil_image, _targets_found, detection_time, frameuuid)
    if len(_targets_found) > 0:
        fire_events(args, pil_image, _targets_found, detection_time, frameuuid)

def save_image(args, img, objects, stamp, frameuuid):
    imgc = img.copy()
    draw = ImageDraw.Draw(img)

    prefix = f"{args.name.lower()}"
    directory = args.directory
    box_save_path = f"{directory}/{prefix}_{stamp}_box_{frameuuid}.jpg"
    nobox_save_path = f"{directory}/{prefix}_{stamp}_nobox_{frameuuid}.jpg"
    box_save_path_latest = (f"{directory}/{prefix}_latest_box.jpg")
    all_box_save_path_latest = (f"{directory}/all_latest_box.jpg")
    nobox_save_path_latest = (f"{directory}/{prefix}_latest_nobox.jpg")
    all_nobox_save_path_latest = (f"{directory}/all_latest_nobox.jpg")

    has_boxes = False
    counter = {}

    for obj in objects:
        has_boxes = True
        inc = 0
        label = obj['name']
        if label in counter:
            inc = counter[label] + 1
        counter[label] = inc
        confidence = obj['confidence']
        model = obj['model']
        prediction_type = 'object'
        box = obj['bounding_box']
        box_area = obj['box_area']
        centroid = obj['centroid']
        predid = obj['uuid']
        imageid = obj['parent_id']
        entity_id = prefix

        crop_save_path = f"{directory}/{prefix}_{stamp}_{model}_{prediction_type}_{label}_{predid}.jpg"
        crop_save_path_pad = f"{directory}/{prefix}_{stamp}_{model}_{prediction_type}_{label}_{predid}_pad.jpg"
        crop_save_path_latest = f"{directory}/{prefix}_latest_{prediction_type}_{label}_{inc}.jpg"
        crop_save_path_latest_pad = f"{directory}/{prefix}_latest_{prediction_type}_crop.jpg"
        all_crop_save_path_latest_pad = f"{directory}/all_latest_{prediction_type}_crop.jpg"
        all_crop_save_path_latest = f"{directory}/all_latest_{prediction_type}_{label}_{inc}.jpg"
        allni_crop_save_path_latest = f"{directory}/all_latest_{prediction_type}_{label}.jpg"
        all_latest_crop = (f"{directory}/all_latest_crop.jpg")

        logger.debug(f"Drawing boxes")
        has_boxes = True
        box_label = f"{model}.{label}: {confidence:.1f}%"
        box_label = f"{obj['model']}.{obj['name']}: {obj['confidence']:.1f}%"
        box_colour = (0, 255, 0)
        draw_box(
            draw,
            (box['y_min'], box['x_min'], box['y_max'], box['x_max']),
            img.width,
            img.height,
            text=box_label,
            color=box_colour,
        )

        draw.text(
            (centroid['x'] * img.width, centroid['y'] * img.height),
            text="X",
            fill=box_colour,
        )

        imc = imgc.crop((box['x_min'] * img.width, box['y_min'] * img.height, box['x_max'] * img.width, box['y_max'] * img.height))

        wmw=368
        wmh=368
        ratio = wmw/wmh
        pad = 1.5
        w, h = img.width, img.height
        x1, y1, x2, y2 = w*box['x_min'], h*box['y_min'], w*box['x_max'], h*box['y_max']
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
        font = ImageFont.truetype("/app/arial.ttf", fz)
        #draw.text((0, 0), datetime.now(EST).strftime("%H:%M"), font=font, fill="#39ff14", stroke_width=2, stroke_fill="#39ff14")
        draw.text((0, 0), datetime.now(pytz.timezone('US/Eastern')).strftime("%H:%M"), font=font, fill="#f93822")

        obj['path'] = f"{crop_save_path}"
        obj['path_crop'] = f"{crop_save_path_pad}"

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
                0,
                label,
                box_area,
                int(box['x_min'] * img.width), int(box['y_min'] * img.height), int(box['x_max'] * img.width), int(box['y_max'] * img.height)
            ))


        logger.debug(f"Saving crops")

        imc.save(crop_save_path)
        imc.save(crop_save_path_latest)
        imc.save(all_crop_save_path_latest)
        imc.save(allni_crop_save_path_latest)
        imcp.save(crop_save_path_pad)
        imcp.save(crop_save_path_latest_pad)
        imcp.save(all_crop_save_path_latest_pad)
        imcp.save(all_latest_crop)

    imgc.save(nobox_save_path_latest)
    imgc.save(nobox_save_path)
    imgc.save(all_nobox_save_path_latest)

    if has_boxes:
        img.save(all_box_save_path_latest)
        img.save(box_save_path_latest)
        img.save(box_save_path)


def fire_events(args, img, objects, stamp, frameuuid):
    global mqtt_client
    logger.debug(f"Firing events")
    for obj in objects:
        prefix = f"{args.name.lower()}"
        directory = args.directory
        crop_save_path = f"{directory}/{prefix}_{stamp}_{obj['model']}_object_{obj['name']}_{obj['uuid']}.jpg"
        timestamp_save_path = f"{directory}/{prefix}_{stamp}_nobox_{frameuuid}.jpg"
        obj['path'] = crop_save_path
        obj['entity_id'] = args.name
        obj['path_frame'] = timestamp_save_path

        message = json.dumps(obj)
        logger.debug(f"Sending to topic {args.mqtt_topic} message {message}")
        mqtt_client.publish(args.mqtt_topic, message, 1)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        #logger.info("Connected to broker")
        global mqtt_connected
        mqtt_connected = True
    else:
        logger.critical("Connection failed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', required=False, type=str, metavar='SN', help='Sensor name', default='default')
    parser.add_argument('--exclude', required=False, type=str, metavar='EX', help='Exclusions', default="/mnt/localshared/data/hassio/tstreamer/exclude.lst")
    parser.add_argument('--directory', required=False, type=str, metavar='CD', help='Storage directory name', default='/mnt/localshared/data/hassio/tstreamer')
    parser.add_argument('--delay', required=False, type=int, default='1000', metavar='D', help='Delay in ms')
    parser.add_argument('--stream', required=False, type=str, metavar='URL', help='stream URL', default='http://192.168.10.246/snap.jpeg')
    parser.add_argument('--triton-url', required=False, type=str, metavar='TRITONIP', help='Triton URL', default='http://192.168.10.23:9050')
    parser.add_argument('--mqtt-ip', required=False, type=str, metavar='MQTTPIP', help='MQTT IP', default='192.168.10.22')
    parser.add_argument('--mqtt-port', required=False, default=1883, type=int, metavar='MQTTPORT', help='MQTT port (default:1883)')
    parser.add_argument('--mqtt-user', required=False, type=str, metavar='MQTTUSR', help='MQTT user', default='xaser')
    parser.add_argument('--mqtt-password', required=False, type=str, metavar='MQTTPWD', help='MQTT password', default='SnetBil8a')
    parser.add_argument('--mqtt-topic', required=False, type=str, default='gs2mqtt', metavar='MQTTTOPIC', help='MQTT Topics')
    parser.add_argument('--debug', required=False, type=str, default='DEBUG', metavar='LEVEL', help='Debug Level')
    parser.add_argument('--min-confidence', required=False, default=60, type=int, metavar='C', help='Minimum confidence (default:60)')
    args = parser.parse_args()

    logger = logging.getLogger(f"stream2mqtt.{args.name}")
    logger.setLevel(args.debug)
    fh = RotatingFileHandler(f'{args.directory}/gs2mqtt.{args.name}.log', maxBytes=(1048576*10), backupCount=7)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Starting streamer")

    mqtt_client = mqttClient.Client("Python")
    mqtt_client.username_pw_set(args.mqtt_user, password=args.mqtt_password)
    mqtt_client.on_connect = on_connect
    mqtt_client.connect(args.mqtt_ip, port=args.mqtt_port)
    mqtt_client.loop_start()

    logger.debug("MQTT loop started")

    i = 1
    loop = True
    while loop:
        if "http" in args.stream:
            frame = Image.open(requests.get(args.stream, stream=True).raw)
        else:
            frame = Image.open(args.stream)
            loop = False
        tic = time.perf_counter()
        try:
            process_image(frame, args)
        except OSError as err:
            logger.debug(f"OS error {err}")
            raise
        except:
            logger.debug(f"Unexpected error {sys.exc_info()}")
            raise
        frame_save_path = f"{args.directory}/{args.name}_last_frame.jpg"
        frame.save(frame_save_path)
        i = i + 1
        time.sleep(args.delay/1000)



