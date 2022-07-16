# This Python file uses the following encoding: utf-8


import os
import sys
from pathlib import Path

import cv2
import torch
# import torch.backends.cudnn as cudnn
import numpy as np
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS
from utils.general import (LOGGER, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync


class YoloV5:
    def __init__(self) -> None:
        self.img_size = 640
        self.stride = 32
        self.weights = ROOT / 'yolov5s.pt'
        self.data = ROOT / 'data/coco128.yaml'
        self.device = 0
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det = 1000
        self.dnn = False
        self.imgsz = (640,640)
        self.augment=False
        self.visualize=False
        self.classes=None
        self.agnostic_nms=False
        self.line_thickness=3
        self.hide_labels=False  # hide labels
        self.hide_conf=True  # hide confidences
        self.view_img=False

    def load_img(self, img0):
        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=True)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        return img

    def select_dev(self, dev):
        self.device = dev

    def load_model(self):
        self.dev = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, self.dev, self.dnn, self.data) 

    def detect(self, srcImg):

        names = self.model.names

        im = self.load_img(srcImg)
        im = torch.from_numpy(im).to(self.dev)
        im = im.float()  # uint8 to fp16/32
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        # Process predictions
        det = pred[0]
        im0 = srcImg.copy()
        annotator = Annotator(im0, line_width=self.line_thickness, example=str(names))
        class_list = []
        conf_list = []
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                class_list.append(names[c])
                conf_list.append(f'{conf:.2f}')
                label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))

        im0 = annotator.result()

        if self.view_img:
            cv2.imshow("img", im0)
            cv2.waitKey(0)  # 1 millisecond

        return im0.copy(),class_list,conf_list



if __name__ == '__main__':
    yolo = YoloV5()
    yolo.load_model()
    img = cv2.imread(str(ROOT / 'data/images/bus.jpg'))
    im0, class_list,conf_list = yolo.detect(img)
    print(class_list,conf_list)
    cv2.imshow("img", im0)
    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()

    
    
    
