"""Floor saw-cut joint detection (thin dark lines on lighter concrete).

Used to (a) sanity-check that joints are isolable from box clutter, and (b) give
each tile a joint-grid PHASE so non-overlapping tiles can be snapped so their
periodic floor joints stay continuous across boundaries.

Method: restrict to the concrete floor (low-saturation, mid-value), enhance thin
dark lines with a morphological black-hat, then take 1-D projection profiles
(columns -> vertical joints, rows -> horizontal joints) and find their peaks.
"""
from __future__ import annotations

import numpy as np
import cv2


def floor_mask(bgr):
    """Low-saturation, mid-value pixels = bare concrete floor (not colored goods)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    m = (s < 55) & (v > 35) & (v < 230)
    m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return m


def joint_response(bgr, valid=None):
    """Black-hat enhanced thin-dark-line response, restricted to the floor."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    fm = floor_mask(bgr)
    if valid is not None:
        fm = cv2.bitwise_and(fm, valid)
    resp = cv2.bitwise_and(bh, bh, mask=fm)
    return resp, fm


def _peaks(profile, min_dist, frac=0.30):
    profile = profile.astype(np.float64)
    if profile.max() <= 0:
        return np.array([], int)
    p = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 9), 0).ravel()
    thr = frac * p.max()
    peaks = []
    i = 1
    while i < len(p) - 1:
        if p[i] >= thr and p[i] >= p[i - 1] and p[i] >= p[i + 1]:
            if not peaks or i - peaks[-1] >= min_dist:
                peaks.append(i)
            elif p[i] > p[peaks[-1]]:
                peaks[-1] = i
        i += 1
    return np.array(peaks, int)


def joint_lines(bgr, valid=None, min_dist=35):
    """Return (vertical_joint_xs, horizontal_joint_ys, response, floor_mask)."""
    resp, fm = joint_response(bgr, valid)
    col_prof = resp.sum(axis=0)      # vertical joints -> peaks in x
    row_prof = resp.sum(axis=1)      # horizontal joints -> peaks in y
    xs = _peaks(col_prof, min_dist)
    ys = _peaks(row_prof, min_dist)
    return xs, ys, resp, fm


if __name__ == "__main__":
    from pathlib import Path
    OUT = Path(__file__).resolve().parent / "out"
    for cam in ["09", "13", "08", "14"]:
        im = cv2.imread(str(OUT / "stills" / f"{cam}.png"))
        if im is None:
            continue
        xs, ys, resp, fm = joint_lines(im)
        vis = im.copy()
        for x in xs:
            cv2.line(vis, (x, 0), (x, im.shape[0]), (0, 0, 255), 2)
        for y in ys:
            cv2.line(vis, (0, y), (im.shape[1], y), (0, 255, 0), 2)
        cv2.imwrite(str(OUT / f"joints_{cam}.png"),
                    np.hstack([vis, cv2.cvtColor(cv2.normalize(resp, None, 0, 255,
                              cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)]))
        print(f"cam {cam}: {len(xs)} vertical joints x={list(xs)}, "
              f"{len(ys)} horizontal joints y={list(ys)}")
