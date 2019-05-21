
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from os import listdir
from os.path import join, isfile

import cv2
import time
import threading
import numpy as np


STD_DIMENSIONS =  {
  "480p": (640, 480),
  "720p": (1280, 720),
  "1080p": (1920, 1080),
  "4k": (3840, 2160),
}

VIDEO_TYPE = {
  'avi': cv2.VideoWriter_fourcc(*'XVID'),
  'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def change_resolution(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

def check_device(source):
    # source: path to the device /dev/
    cap = cv2.VideoCapture(source)
    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', source)

# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    # change the current caputre device to the resulting resolution
    # otherwise, set the lowest.
    change_resolution(cap, width, height)
    return width, height

class RealtimeCapture(object):
    def __init__(self, capid):
        self._frame = None
        self._frame_idx = 0
        self._status = None
        self._isstop = False
        self._capid = capid

        self._capture = cv2.VideoCapture(self._capid)
        get_dims(self._capture, res="720p")
        self.start()
        time.sleep(1)

    def start(self):
        time.sleep(1)
        threading.Thread(target=self._queryframe, daemon=True, args=()).start()

    def stop(self):
        self._isstop = True

    def read (self):
        return self._frame

    def release(self):
        self.stop()
        self._capture.release()

    def _queryframe(self):
        while(not self._isstop):
            self._status, self._frame = self._capture.read()
            self._frame_idx += 1
        self._capture.release()