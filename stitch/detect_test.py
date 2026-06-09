"""Quick validation: does YOLO-World-S detect pallets/boxes at near-nadir on our plates?

Tests text prompts at low confidence on a few cameras (open floor + packed). If text
recall is poor (expected at top-down per the research), we fall back to visual prompts.

Run:  ./venv/bin/python stitch/detect_test.py
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
PROMPTS = ["wooden pallet", "stacked cardboard boxes", "shrink-wrapped pallet",
           "stack of boxes", "palletized goods"]
CAMS = ["07", "08", "01", "03", "13", "16"]
CONF = 0.04


def main():
    from ultralytics import YOLOWorld
    m = YOLOWorld("yolov8s-worldv2.pt")
    m.set_classes(PROMPTS)
    print(f"prompts={PROMPTS}\nconf={CONF}\n")
    for cam in CAMS:
        f = OUT / "plates" / f"{cam}.png"
        if not f.exists():
            continue
        img = cv2.imread(str(f))
        r = m.predict(img, conf=CONF, iou=0.5, verbose=False)[0]
        n = 0 if r.boxes is None else len(r.boxes)
        vis = img.copy()
        if r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                cls = int(b.cls[0]); sc = float(b.conf[0])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(vis, ((x1 + x2) // 2, y2), 5, (0, 0, 255), -1)  # footprint = box bottom-center
                cv2.putText(vis, f"{PROMPTS[cls][:10]} {sc:.2f}", (x1, max(y1 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(OUT / f"detect_{cam}.png"), vis)
        print(f"  cam {cam}: {n} detections")
    print(f"\n  saved detect_<cam>.png to stitch/out/")


if __name__ == "__main__":
    main()
