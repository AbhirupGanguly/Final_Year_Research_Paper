"""
main.py
-------
Entry point for Detection + Tracking only.

Usage:
    python main.py                          # uses input_video.mp4 by default
    python main.py --video path/to/vid.mp4  # custom video path
    python main.py --no-display             # headless (saves output only)

Output:
    Annotated live window  (press Q to quit)
    output_annotated.mp4   (saved next to this script)
"""

import cv2
import argparse
import os
import sys

from detector import PersonDetector
from tracker import IOUTracker
from visualizer import draw_tracks, draw_info


# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_VIDEO = "input_video.mp4"
DEFAULT_MODEL = "yolov8n.pt"
OUTPUT_FILE   = "output_annotated.mp4"
PROCESS_EVERY_N_FRAMES = 1   # set to 2 or 3 to skip frames for speed


def parse_args():
    parser = argparse.ArgumentParser(description="Person Detection + Tracking")
    parser.add_argument("--video",      default=DEFAULT_VIDEO,
                        help="Path to input video (default: input_video.mp4)")
    parser.add_argument("--model",      default=DEFAULT_MODEL,
                        help="Path to YOLOv8 weights (default: yolov8n.pt)")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable live window (useful on headless servers)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── validate inputs ──────────────────────────────────────────────────────
    if not os.path.isfile(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        print("        Place a video named 'input_video.mp4' in this folder,")
        print("        or pass --video <path>")
        sys.exit(1)


    # ── initialise components ────────────────────────────────────────────────
    detector = PersonDetector(model_path=args.model)
    tracker  = IOUTracker()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (width, height))

    print(f"[INFO] Video  : {args.video}  ({width}x{height} @ {fps:.1f} fps, {total} frames)")
    print(f"[INFO] Model  : {args.model}")
    print(f"[INFO] Output : {OUTPUT_FILE}")
    print("[INFO] Press Q in the window to stop early.\n")

    frame_number = 0
    last_tracks  = []          # reuse previous tracks on skipped frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        # ── detection + tracking (every N frames) ────────────────────────
        if frame_number % PROCESS_EVERY_N_FRAMES == 0:
            detections  = detector.detect(frame)
            last_tracks = tracker.update(detections)
            count = len(last_tracks)
            print(f"[FRAME {frame_number:>5}] People detected: {count}")

        # ── draw results ─────────────────────────────────────────────────
        draw_tracks(frame, last_tracks)
        draw_info(frame, frame_number, len(last_tracks))

        writer.write(frame)

        if not args.no_display:
            cv2.imshow("Detection + Tracking  (Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Stopped by user.")
                break

    # ── cleanup ──────────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\n[DONE] Processed {frame_number} frames.")
    print(f"[DONE] Saved annotated video to: {OUTPUT_FILE}")
    print(f"[DONE] Unique people detected in video: {tracker.total_unique}")


if __name__ == "__main__":
    main()
