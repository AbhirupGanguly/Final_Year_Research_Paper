"""
visualizer.py
-------------
Draws bounding boxes and TEMPORARY tracking IDs onto video frames.
No names, roll numbers, or identity info — only numeric track IDs.
"""

import cv2


# ── colour palette (cycles if more than 20 tracks) ──────────────────────────
_PALETTE = [
    (255,  56,  56), (255, 157,  75), ( 75, 255, 150), ( 56, 130, 255),
    (220,  80, 220), ( 75, 220, 255), (255, 255,  75), (200, 255, 100),
    (100, 200, 255), (255, 100, 200), (180, 130, 255), (130, 255, 180),
    (255, 180, 130), (130, 180, 255), (255, 130, 180), ( 80, 255, 255),
    (255,  80, 130), (130, 255, 130), (255, 200, 200), (200, 200, 255),
]

BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.65
FONT_THICKNESS = 2
LABEL_PAD = 6    # pixels of vertical padding for the label background


def _color(track_id: int) -> tuple:
    return _PALETTE[(track_id - 1) % len(_PALETTE)]


def draw_tracks(frame, tracks: list) -> None:
    """
    Draw bounding boxes and track-ID labels directly onto `frame` (in-place).

    Args:
        frame:  BGR image (numpy array).
        tracks: list of (track_id, x1, y1, x2, y2).
    """
    for (track_id, x1, y1, x2, y2) in tracks:
        color = _color(track_id)
        label = f"ID {track_id}"

        # bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # label background
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        bg_y1 = max(y1 - th - LABEL_PAD * 2, 0)
        bg_y2 = max(y1, th + LABEL_PAD * 2)
        cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + LABEL_PAD, bg_y2), color, -1)

        # label text
        text_y = max(y1 - LABEL_PAD, th + LABEL_PAD)
        cv2.putText(frame, label, (x1 + LABEL_PAD // 2, text_y),
                    FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)


def draw_info(frame, frame_number: int, total_tracks: int) -> None:
    """Overlay frame counter and live-person count (top-left corner)."""
    info = f"Frame: {frame_number}  |  Persons: {total_tracks}"
    cv2.putText(frame, info, (10, 30), FONT, 0.6, (200, 200, 200), 2)
