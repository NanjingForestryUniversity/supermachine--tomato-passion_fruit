import os
import sys
from pathlib import Path
import torch
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.segment.general import process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode
import cv2
import numpy as np
from config import Config as setting

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # This function remains the same as defined previously
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, ratio, (dw, dh)
def process_image(img0, img_size=640, stride=32, auto=True):
    """
    Processes an image by resizing and padding it to the required size.

    Args:
    - img0 (np.array): Original image as a numpy array.
    - img_size (int, optional): Desired size of the image. Defaults to 640.
    - stride (int, optional): Stride size for padding. Defaults to 32.
    - auto (bool, optional): If True, automatically adjusts padding to meet stride requirements. Defaults to True.

    Returns:
    - np.array: The processed image ready for model input.
    """
    # Resize and pad the image
    im, _, _ = letterbox(img0, new_shape=img_size, stride=stride, auto=auto)

    # Convert image from HWC to CHW format and from BGR to RGB
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    return im

class TOSEG:
    def __init__(self, weights=Path(setting.toseg_weights), device='', dnn=False, data=None, half=False, imgsz=(640, 640)):
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)
        self.model.warmup(imgsz=(1 if self.pt else 1, 3, *self.imgsz))

    @smart_inference_mode()
    #返回结果图像
    # def visualize(self, image, results, line_thickness=3, hide_labels=False, hide_conf=False):
    #     annotator = Annotator(image, line_width=line_thickness)
    #
    #     # 获取原始图像尺寸
    #     h, w = image.shape[:2]
    #
    #     # 准备 im_gpu 参数
    #     im_gpu = torch.as_tensor(image, dtype=torch.float16, device=self.device).permute(2, 0, 1).flip(
    #         0).contiguous() / 255
    #
    #     for r in results:
    #         box = r['xyxy']
    #         leaf = r['leaf']
    #         label = None if hide_labels else (r['label'] if hide_conf else f"{r['label']} {r['conf']:.2f}")
    #         annotator.box_label(box, label, color=colors(r['cls'], True))
    #
    #         # 确保 leaf 是正确的格式并调整大小
    #         if isinstance(leaf, np.ndarray):
    #             leaf = torch.from_numpy(leaf).to(self.device)
    #         elif isinstance(leaf, list):
    #             leaf = torch.tensor(leaf, device=self.device)
    #
    #         # 如果 leaf 是 2D，添加批次维度
    #         if leaf.ndim == 2:
    #             leaf = leaf.unsqueeze(0)
    #
    #         # 调整掩码大小以匹配原始图像
    #         leaf = torch.nn.functional.interpolate(leaf.unsqueeze(1).float(), size=(h, w), mode='bilinear',
    #                                                align_corners=False).squeeze(1)
    #
    #         annotator.masks(leaf, colors=[colors(r['cls'], True)], im_gpu=im_gpu)
    #
    #     return annotator.result()
    #返回掩码图像
    def visualize(self, image, results, line_thickness=3, hide_labels=False, hide_conf=False):
        # 创建一个全白的背景图像
        background = np.zeros_like(image) * 255  # 将背景设置为白色

        h, w = image.shape[:2]  # 获取图像尺寸

        for r in results:
            mask = r['leaf']
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).to(self.device)
            elif isinstance(mask, list):
                mask = torch.tensor(mask, device=self.device)

            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            # 调整掩码大小以匹配原始图像
            mask = torch.nn.functional.interpolate(mask.unsqueeze(1).float(), size=(h, w), mode='bilinear',
                                                   align_corners=False).squeeze(1)

            # 将遮罩应用于背景
            black_mask = (mask.cpu().numpy() > 0.5)  # 创建一个黑色遮罩
            for i in range(3):  # 对每个颜色通道进行操作
                background[:, :, i] = np.where(black_mask, 255, background[:, :, i])  # 在遮罩区域应用黑色

        return background

    def predict(self, source, conf_thres=0.25, iou_thres=0.45, max_det=1000, classes=None,
                agnostic_nms=False, augment=False, retina_masks=False):
        # dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)


        # for path, im, im0s, vid_cap, s in dataset:
            im0s = source
            im = process_image(im0s, img_size=self.imgsz, stride=self.stride, auto=self.pt)
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            pred, proto = self.model(im, augment=augment)[:2]
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

            results = []
            for i, det in enumerate(pred):  # per image
                if len(det):
                    if retina_masks:
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
                        masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0s.shape[:2])
                    else:
                        masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()

                    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                        c = int(cls)
                        label = f'{self.names[c]} {conf:.2f}'
                        mask = masks[j]
                        results.append({
                            'xyxy': xyxy,
                            'conf': conf,
                            'cls': c,
                            'label': label,
                            'leaf': mask
                        })

            return results, im0s

    def toseg(self, img):
        results, image = self.predict(img)
        vaa = self.visualize(image, results)
        mask = cv2.cvtColor(vaa, cv2.COLOR_RGB2GRAY)
        return mask