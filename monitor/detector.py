"""Detector de vehículos YOLO (ONNX) sin dependencias de torch."""
from __future__ import annotations

import io

import numpy as np
import onnxruntime as ort
from PIL import Image

# Clases COCO de vehículos: car, motorcycle, bus, truck
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class VehicleDetector:
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.5):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2])
        self.conf = conf
        self.iou = iou

    def _letterbox(self, img: Image.Image):
        w, h = img.size
        r = self.size / max(w, h)
        nw, nh = round(w * r), round(h * r)
        canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        px, py = (self.size - nw) // 2, (self.size - nh) // 2
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
        x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        return x, r, px, py

    def _run(self, img: Image.Image, roi=None):
        """Detecta en la región roi (x0,y0,x1,y1 normalizada). Devuelve (xyxy en px de img, scores, clases)."""
        W, H = img.size
        ox = oy = 0
        if roi is not None:
            ox, oy = int(roi[0] * W), int(roi[1] * H)
            img = img.crop((ox, oy, int(roi[2] * W), int(roi[3] * H)))
        w, h = img.size
        x, r, px, py = self._letterbox(img)
        out = self.session.run(None, {self.input_name: x})[0][0].T  # (N, 84)
        cls_scores = out[:, 4:]
        cls_ids = cls_scores.argmax(1)
        scores = cls_scores[np.arange(len(cls_ids)), cls_ids]
        keep = (scores >= self.conf) & np.isin(cls_ids, list(VEHICLE_CLASSES))
        boxes, scores, cls_ids = out[keep, :4], scores[keep], cls_ids[keep]
        cx, cy, bw, bh = boxes.T
        xyxy = np.stack([(cx - bw / 2 - px) / r, (cy - bh / 2 - py) / r,
                         (cx + bw / 2 - px) / r, (cy + bh / 2 - py) / r], 1).clip(0, [w, h, w, h])
        xyxy += [ox, oy, ox, oy]
        return xyxy.reshape(-1, 4), scores, cls_ids

    def detect(self, image_bytes: bytes, rois=()):
        """Devuelve (detecciones normalizadas 0-1, ancho, alto, brillo medio).

        rois: regiones extra (normalizadas) que se analizan ampliadas, p. ej. la zona de plazas.
        """
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        brightness = float(np.asarray(img.convert("L").resize((64, 36))).mean())
        parts = [self._run(img)] + [self._run(img, roi) for roi in rois]
        xyxy = np.concatenate([p[0] for p in parts])
        scores = np.concatenate([p[1] for p in parts])
        cls_ids = np.concatenate([p[2] for p in parts])
        if not len(scores):
            return [], w, h, brightness
        # NMS agnóstico de clase: un coche detectado como car+truck cuenta una vez
        idx = nms(xyxy, scores, self.iou)
        dets = []
        for i in idx:
            a, b, c, d = xyxy[i]
            dets.append({
                "box": [round(a / w, 4), round(b / h, 4), round(c / w, 4), round(d / h, 4)],
                "conf": round(float(scores[i]), 3),
                "cls": VEHICLE_CLASSES[int(cls_ids[i])],
            })
        return dets, w, h, brightness


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float):
    order = scores.argsort()[::-1]
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < thr]
    return keep
