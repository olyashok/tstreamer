#!/usr/bin/env python

import cv2
import gi
import numpy as np
import time

gi.require_version('Gst', '1.0')
from gi.repository import Gst


class Video():
    def __init__(self):
        Gst.init(None)
        self._frame = None

        self.video_source = 'rtsp://192.168.10.1:7447/ahoCPkXFexpDjXAS'
        self.video_codec = '! rtph264depay ! h264parse ! nvh264dec'
        #self.video_decode = '! decodebin ! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert'
        self.video_decode = '! video/x-raw ! videoconvert'
        self.video_sink_conf = '! appsink emit-signals=true sync=false max-buffers=2 drop=true'

        self.video_pipe = None
        self.video_sink = None

        self.run()

    def start_gst(self, config=None):
        if not config:
            config = \
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
            e = w*h
            Y = np.frombuffer(byteArray[0:e],dtype=np.uint8)
            Y = np.reshape(Y, (h,w))

            s = e
            V = np.frombuffer(byteArray[s::2],dtype=np.uint8)
            V = np.repeat(V, 2, 0)
            V = np.reshape(V, (int(h/2),w))
            V = np.repeat(V, 2, 0)

            U = np.frombuffer(byteArray[s+1::2],dtype=np.uint8)
            U = np.repeat(U, 2, 0)
            U = np.reshape(U, (int(h/2),w))
            U = np.repeat(U, 2, 0)

            RGBMatrix = (np.dstack([Y,U,V])).astype(np.uint8)
            RGBMatrix = cv2.cvtColor(RGBMatrix, cv2.COLOR_YUV2RGB, 3)
        return RGBMatrix

    def frame(self):
        return self._frame

    def frame_available(self):
        return type(self._frame) != type(None)

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


if __name__ == '__main__':
    video = Video()
    i = 1
    while True:
        # Wait for the next frame
        if not video.frame_available():
            continue

        frame = video.frame()
        cv2.imwrite(f"/mnt/nas_downloads/data/gstest/frame{i}.jpg", frame)
        print("saved frame")

        time.sleep(10)
        i = i+1

        if i>10:
            quit()