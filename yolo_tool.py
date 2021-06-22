from __future__ import print_function

import cv2
import numpy as np
import os
import wget
import tarfile

import pycuda.driver as cuda

from data_processing import load_label_categories

FPS = 30
GST_STR_CSI = "nvarguscamerasrc \
    ! video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1, sensor-id=%d \
    ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx \
    ! videoconvert \
    ! appsink"
WINDOW_NAME = "Tiny YOLO v2"
INPUT_RES = (416, 416)
# MODEL_URL = 'https://onnxzoo.blob.core.windows.net/models/opset_8/tiny_yolov2/tiny_yolov2.tar.gz'
MODEL_URL = "https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.tar.gz"
LABEL_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/voc.names"


# Draw bounding boxes on the screen from the YOLO inference result
def draw_bboxes(image, bboxes, confidences, categories, all_categories, message=None):
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        img_height, img_width, _ = image.shape
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(img_width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(img_height, np.floor(y_coord + height + 0.5).astype(int))
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 3)
        info = "{0} {1:.2f}".format(all_categories[category], score)
        cv2.putText(
            image,
            info,
            (right, top),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        print(info)
    if message is not None:
        cv2.putText(
            image,
            message,
            (32, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )


# Draw the message on the screen
def draw_message(image, message):
    cv2.putText(
        image,
        message,
        (32, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )


# Reshape the image from OpneCV to Tiny YOLO v2
def reshape_image(img):
    # Convert 8-bit integer to 32-bit floating point
    img = img.astype(np.float32)
    # Convert HWC to CHW
    img = np.transpose(img, [2, 0, 1])
    # Convert CHW to NCHW
    img = np.expand_dims(img, axis=0)
    # Convert to row-major
    img = np.array(img, dtype=np.float32, order="C")
    return img


# Download file from the URL if it doesn't exist yet.
def download_file_from_url(url):
    file = os.path.basename(url)
    if not os.path.exists(file):
        print("\nDownload from %s" % url)
        wget.download(url)
    return file


# Download the label file if it doesn't exist yet.
def download_label():
    file = download_file_from_url(LABEL_URL)
    categories = load_label_categories(file)
    num_categories = len(categories)
    assert num_categories == 20
    return categories


# Download the Tiny YOLO v2 ONNX model file and extract it
# if it doesn't exist yet.
def download_model():
    file = download_file_from_url(MODEL_URL)
    tar = tarfile.open(file)
    infs = tar.getmembers()
    onnx_file = None
    for inf in infs:
        f = inf.name
        _, ext = os.path.splitext(f)
        if ext == ".onnx":
            onnx_file = f
            break
    if not os.path.exists(onnx_file):
        tar.extract(onnx_file)
    tar.close()
    return onnx_file
