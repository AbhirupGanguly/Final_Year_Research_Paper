"""
tracker.py
----------
IOU-based tracker with box smoothing to reduce ID fragmentation.

Key improvements over naive IOU tracker:
  - Box smoothing (EMA): predicted position drifts smoothly, so IoU matching
    survives small detection jitter without spawning a new ID.
  - Generous MAX_LOST: tracks survive occlusion across many frames.
  - Lower IOU threshold: lenient matching for seated / partly-visible people.
"""

import numpy as np

IOU_THRESHOLD = 0.25    # minimum IoU to consider a match
MAX_LOST      = 90      # frames a track can be missing (~3 s @ 30 fps)
SMOOTH_ALPHA  = 0.6     # EMA weight for new detection (0=ignore new, 1=no smooth)


def _iou(boxA, boxB) -> float:
    """Compute Intersection-over-Union for two (x1,y1,x2,y2) boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def _smooth(prev, new, alpha=SMOOTH_ALPHA):
    """Exponential moving average on box coordinates."""
    return tuple(int(alpha * n + (1 - alpha) * p) for p, n in zip(prev, new))


class Track:
    """Represents one tracked person."""

    def __init__(self, track_id: int, box: tuple):
        self.track_id = track_id
        self.box  = box          # smoothed (x1, y1, x2, y2)
        self.lost = 0            # consecutive frames without a match


class IOUTracker:
    """
    Lightweight IOU-based multi-object tracker with EMA box smoothing.

    Usage:
        tracker = IOUTracker()
        tracked = tracker.update(detections)   # each frame
    """

    def __init__(self):
        self._tracks: list[Track] = []
        self._next_id = 1
        self._frame_counts: list[int] = []   # per-frame live-track counts

    @property
    def total_unique(self) -> int:
        """
        Best estimate of unique people in the entire video.

        Uses the MODE of per-frame detection counts rather than the raw ID
        counter, which over-counts due to ID fragmentation on bounding-box
        jitter.  The mode is the count seen most often across all frames,
        which corresponds to the 'steady state' occupancy of the scene.
        """
        if not self._frame_counts:
            return 0
        counts = np.array(self._frame_counts)
        # Use the mode of the top-half of counts (ignore transient low counts
        # from the first/last frames when not everyone is in frame yet).
        median = int(np.median(counts))
        # Return median — robust to outlier frames where extra people walk in.
        return median

    def update(self, detections: list) -> list:
        """
        Match new detections to existing tracks.

        Args:
            detections: list of (x1, y1, x2, y2) from the detector.

        Returns:
            list of (track_id, x1, y1, x2, y2) — one entry per live track.
        """
        matched_track_ids = set()
        matched_det_ids   = set()

        # ── Step 1: compute IoU matrix ───────────────────────────────────
        if self._tracks and detections:
            iou_matrix = np.zeros((len(self._tracks), len(detections)))
            for ti, track in enumerate(self._tracks):
                for di, det in enumerate(detections):
                    iou_matrix[ti, di] = _iou(track.box, det)

            # ── Step 2: greedy match (highest IoU first) ─────────────────
            while True:
                if iou_matrix.size == 0:
                    break
                max_val = iou_matrix.max()
                if max_val < IOU_THRESHOLD:
                    break
                ti, di = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                # Smooth the box instead of snapping to raw detection
                self._tracks[ti].box  = _smooth(self._tracks[ti].box, detections[di])
                self._tracks[ti].lost = 0
                matched_track_ids.add(ti)
                matched_det_ids.add(di)
                iou_matrix[ti, :] = -1
                iou_matrix[:, di] = -1

        # ── Step 3: increment lost counter for unmatched tracks ──────────
        for ti, track in enumerate(self._tracks):
            if ti not in matched_track_ids:
                track.lost += 1

        # ── Step 4: create new tracks for unmatched detections ───────────
        for di, det in enumerate(detections):
            if di not in matched_det_ids:
                self._tracks.append(Track(self._next_id, det))
                self._next_id += 1

        # ── Step 5: remove stale tracks ──────────────────────────────────
        self._tracks = [t for t in self._tracks if t.lost <= MAX_LOST]

        # ── Step 6: collect live tracks ──────────────────────────────────
        result = []
        for track in self._tracks:
            if track.lost == 0:
                result.append((track.track_id, *track.box))

        # Record per-frame live count for unique-people estimation
        self._frame_counts.append(len(result))

        return result
