import cv2

from utils import Singleton

gstreamer_pipeline = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! appsink"


class Camera(Singleton):
    def __init__(self):
        self.video = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

    def __del__(self):
        self.video.release()

    def get_image(self):
        success, image = self.video.read()
        return success, image
