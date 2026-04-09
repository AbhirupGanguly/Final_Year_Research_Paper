"""
detector.py
-----------
Detects PEOPLE in a single frame using YOLOv8 (class 0 = 'person').
Returns a list of bounding boxes [(x1, y1, x2, y2), ...].

Optimisations over the old Haar/MTCNN approach
-----------------------------------------------
- Uses the native ultralytics YOLO inference pipeline (batched, GPU-friendly).
- Runs at a configurable input resolution (IMG_SIZE) — smaller = faster.
- Half-precision (fp16) inference when a CUDA GPU is available.
- Filters only class-0 ("person") detections.
- Configurable confidence and IOU-NMS thresholds.
"""

import cv2
import torch

from ultralytics import YOLO

# ── Tuning constants ─────────────────────────────────────────────────────────
IMG_SIZE    = 1280   # Larger resolution catches distant/small people in classrooms.
                     # Drop to 960 or 640 if too slow.
CONF_THRESH = 0.20   # Low threshold — catches seated, partial, and far-away people.
                     # Raise to 0.30 if you get too many false positives.
IOU_THRESH  = 0.40   # NMS IoU — moderate overlap allowed for adjacent seats.
PERSON_CLS  = 0      # COCO class index for "person"
AUGMENT     = True   # Test-time augmentation: flips + scales → better recall.
                     # Set False to save ~30% time at slight accuracy cost.

# Use GPU half-precision when available — biggest single speed win
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_HALF   = _DEVICE == "cuda"


class PersonDetector:
    """
    YOLOv8-based person detector.

    Args:
        model_path: Path to YOLOv8 weights file (default: yolov8n.pt).
                    yolov8n.pt  — fastest  (nano)
                    yolov8s.pt  — balanced (small)   ← recommended for classrooms
                    yolov8m.pt  — accurate (medium)
        img_size:   Inference resolution.  Smaller = faster, less accurate.
        conf:       Confidence threshold.
        iou:        NMS IoU threshold.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        img_size: int   = IMG_SIZE,
        conf: float     = CONF_THRESH,
        iou: float      = IOU_THRESH,
    ):
        self._img_size = img_size
        self._conf     = conf
        self._iou      = iou

        print(f"[PersonDetector] Loading YOLOv8 model: {model_path}")
        print(f"[PersonDetector] Device : {_DEVICE.upper()}  |  half-precision: {_HALF}")
        print(f"[PersonDetector] Conf   : {conf}  |  IoU NMS: {iou}  |  img_size: {img_size}  |  augment: {AUGMENT}")

        self._model = YOLO(model_path)
        self._model.to(_DEVICE)

    # ── public API ───────────────────────────────────────────────────────────

    def detect(self, frame) -> list:
        """
        Run person detection on one BGR frame.

        Returns:
            List of (x1, y1, x2, y2) integer tuples, one per detected person.
            Boxes are clipped to frame boundaries.
        """
        h, w = frame.shape[:2]

        results = self._model.predict(
            source=frame,
            imgsz=self._img_size,
            conf=self._conf,
            iou=self._iou,
            classes=[PERSON_CLS],   # only "person", no other classes
            half=_HALF,
            device=_DEVICE,
            augment=AUGMENT,        # test-time augmentation for better recall
            verbose=False,          # suppress per-frame YOLO console spam
        )

        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w, int(x2))
                y2 = min(h, int(y2))
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))

        return boxes
