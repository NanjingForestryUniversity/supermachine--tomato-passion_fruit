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

@smart_inference_mode()
def run(
    img,  # numpy array
    weights=Path(r'D:\porject\PY\20240627Actual_deployed\weights\best.pt'),  # model path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="0",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    half=False,  # use FP16 half-precision inference
):
    """Runs YOLOv5 detection inference on a numpy array and returns the number of detections."""
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, fp16=half)
    stride = model.stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Convert numpy array to tensor
    img = letterbox(img, imgsz, stride=stride)[0]
    img = img.transpose((2,0,1))
    img = np.ascontiguousarray(img)
    im = torch.from_numpy(img).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Inference
    pred = model(im)

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)

    # Count detections
    num_detections = sum([len(d) for d in pred if d is not None])

    return num_detections

