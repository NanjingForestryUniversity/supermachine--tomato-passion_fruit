import cv2
import numpy as np
import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_boxes,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode
from utils.dataloaders import letterbox
from config import Config as setting

class Detector_to:
    def __init__(self, weights=Path(setting.tomato_model_path),
                 device="", half=False):
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, fp16=half)
        self.stride = int(self.model.stride)  # get stride from the model
        self.fp16 = half

    def run(self, img, imgsz=(640, 640), conf_thres=0.25, iou_thres=0.45, max_det=1000):
        """Runs YOLOv5 detection inference on a numpy array and returns the number of detections."""
        imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        # Convert numpy array to tensor
        img = letterbox(img, imgsz, stride=self.stride)[0]  # resize image to model expected size
        img = img.transpose((2, 0, 1)) # HWC to CHW
        img = np.ascontiguousarray(img) # make contiguous
        im = torch.from_numpy(img).to(self.model.device)
        im = im.half() if self.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = self.model(im)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)

        # Count detections
        num_detections = sum([len(d) for d in pred if d is not None])

        return num_detections

