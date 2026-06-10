"""Phase 6 — pallet detection behind a pluggable interface.

The whole pipeline depends only on `Detector.detect(frame) -> list[Detection]`, so the
user's existing instance-segmentation model swaps in by replacing the backend, with no
change to footprint extraction, de-dup, or display.

Default backend: YOLO-World-S (open-vocabulary, zero training) — prompt it with
descriptive pallet/box phrases. Footprint = box bottom-centre (or mask base if masks
are present), the floor-contact point that survives parallax on tall stacks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import box_base_point, mask_base_point


@dataclass
class Detection:
    box_xyxy: tuple[float, float, float, float]
    score: float
    label: str
    mask: np.ndarray | None = None        # bool HxW in the frame's pixel space (optional)

    @property
    def footprint(self) -> tuple[float, float]:
        """Floor-contact point (pixels): mask base if available, else box bottom-centre."""
        if self.mask is not None and self.mask.any():
            return mask_base_point(self.mask)
        return box_base_point(self.box_xyxy)


class Detector:
    """Contract: detect(frame_bgr, cam=None) -> list[Detection], coords in pixel space.
    `cam` is optional context (e.g. for per-camera visual prompts)."""
    def detect(self, frame_bgr: np.ndarray, cam: str | None = None) -> list[Detection]:
        raise NotImplementedError


def _yoloe_to_dets(r, frame_shape, labels):
    """Convert an ultralytics result (boxes + masks) into Detection objects."""
    import cv2
    out = []
    if r.boxes is None:
        return out
    H, W = frame_shape[:2]
    md = r.masks.data.cpu().numpy() if r.masks is not None else None
    for i, b in enumerate(r.boxes):
        xyxy = tuple(float(v) for v in b.xyxy[0].cpu().numpy())
        ci = int(b.cls[0])
        label = labels[ci] if labels and ci < len(labels) else "pallet"
        mask = None
        if md is not None and i < len(md):
            mk = md[i]
            if mk.shape != (H, W):
                mk = cv2.resize(mk.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            mask = mk.astype(bool)
        out.append(Detection(xyxy, float(b.conf[0]), label, mask=mask))
    return out


class YOLOWorldDetector(Detector):
    """Open-vocabulary box detector (no training). Lightweight; CPU-friendly with OpenVINO."""

    def __init__(self, prompts: list[str], conf: float = 0.05, iou: float = 0.5,
                 model: str = "yolov8s-worldv2.pt"):
        from ultralytics import YOLOWorld           # lazy: heavy import only when used
        self.model = YOLOWorld(model)
        self.model.set_classes(prompts)
        self.prompts, self.conf, self.iou = prompts, conf, iou

    def detect(self, frame_bgr: np.ndarray, cam: str | None = None) -> list[Detection]:
        r = self.model.predict(frame_bgr, conf=self.conf, iou=self.iou, verbose=False)[0]
        out: list[Detection] = []
        if r.boxes is not None:
            for b in r.boxes:
                xyxy = tuple(float(v) for v in b.xyxy[0].cpu().numpy())
                out.append(Detection(xyxy, float(b.conf[0]), self.prompts[int(b.cls[0])]))
        return out


class YOLOESegDetector(Detector):
    """Open-vocabulary INSTANCE-SEGMENTATION (no training): masks + text OR visual prompts.

    - text mode:   set descriptive class prompts (better recall than YOLO-World here).
    - visual mode: give one+ example pallet box(es) per camera (artifacts/visual_prompts.json);
                   the model matches that camera's actual content (best for hard cameras).
    Masks let the footprint be the mask base (parallax-safe), via Detection.footprint.
    """

    def __init__(self, prompts, conf=0.10, iou=0.5, model="yoloe-11s-seg.pt",
                 mode="text", visual_prompts=None):
        from ultralytics import YOLOE
        self.model = YOLOE(model)
        self.prompts, self.conf, self.iou, self.mode = prompts, conf, iou, mode
        self.vp = visual_prompts or {}        # cam -> list of [x1,y1,x2,y2]
        self._text_pe = self.model.get_text_pe(prompts)
        self.model.set_classes(prompts, self._text_pe)
        self._last_visual = False

    def detect(self, frame_bgr: np.ndarray, cam: str | None = None) -> list[Detection]:
        # HYBRID: visual prompt where this camera has marked examples, else text.
        if self.mode == "visual" and cam in self.vp and self.vp[cam]:
            from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
            vp = dict(bboxes=np.array(self.vp[cam], dtype=float),
                      cls=np.zeros(len(self.vp[cam]), dtype=int))
            r = self.model.predict(frame_bgr, visual_prompts=vp, refer_image=frame_bgr,
                                   predictor=YOLOEVPSegPredictor, conf=self.conf,
                                   iou=self.iou, verbose=False)[0]
            self._last_visual = True
            return _yoloe_to_dets(r, frame_bgr.shape, ["pallet"])
        if self._last_visual:                 # restore text classes after a visual call
            self.model.set_classes(self.prompts, self._text_pe)
            self._last_visual = False
        r = self.model.predict(frame_bgr, conf=self.conf, iou=self.iou,
                               retina_masks=True, verbose=False)[0]
        return _yoloe_to_dets(r, frame_bgr.shape, self.prompts)


class UserSegDetector(Detector):
    """Adapter stub for the user's existing instance-segmentation model.

    Implement __init__ to load weights / open the endpoint, and detect() to return
    Detection objects WITH masks (mask in the frame's pixel space). Everything
    downstream (footprint, de-dup, display) then works unchanged.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Plug the existing instance-seg model in here: load it in __init__, and in "
            "detect(frame) return [Detection(box_xyxy, score, label, mask=<bool HxW>), ...].")

    def detect(self, frame_bgr: np.ndarray, cam: str | None = None) -> list[Detection]:  # pragma: no cover
        raise NotImplementedError


def make_detector(cfg) -> Detector:
    """Build the configured detector. Default: YOLOE-seg (masks + better recall)."""
    import json
    d = cfg.raw.get("detect", {}) if hasattr(cfg, "raw") else {}
    backend = d.get("backend", "yoloeseg")
    prompts = d.get("prompts", ["wooden pallet", "stacked cardboard boxes",
                                "shrink-wrapped pallet", "stack of boxes", "palletized goods"])
    if backend == "yoloeseg":
        vp = {}
        vp_path = cfg.path("artifacts") / "visual_prompts.json"
        if d.get("mode", "text") == "visual" and vp_path.exists():
            vp = json.loads(vp_path.read_text())
        return YOLOESegDetector(prompts=prompts, conf=float(d.get("conf", 0.10)),
                                iou=float(d.get("iou", 0.5)),
                                model=d.get("model", "yoloe-11s-seg.pt"),
                                mode=d.get("mode", "text"), visual_prompts=vp)
    if backend == "yoloworld":
        return YOLOWorldDetector(prompts=prompts, conf=float(d.get("conf", 0.05)),
                                 iou=float(d.get("iou", 0.5)),
                                 model=d.get("model", "yolov8s-worldv2.pt"))
    if backend == "userseg":
        return UserSegDetector(**d.get("userseg", {}))
    raise ValueError(f"unknown detect.backend: {backend}")
