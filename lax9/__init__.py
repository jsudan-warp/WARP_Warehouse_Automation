"""LAX-9 Stage 1 — pallet localization from ceiling cameras.

Pipeline (run via `python -m lax9 <phase>`):
    intake    Phase 0  probe videos, confirm feed count/resolution/fps/length
    frames    Phase 1  one floor-dominant still per camera (median or sharp frame)
    overlaps  Phase 2  pairwise floor matching -> overlap matrix -> neighbour graph
    calibrate Phase 3  lens intrinsics (GATE: checkerboard photos)
    stitch    Phase 4  per-camera floor homography -> shared mosaic
    scale     Phase 5  metric scale & world frame (GATE: measured floor points)
    detect    Phase 6  pallet detector + footprint projection (GATE: pallet model)
    map       Phase 7  metric pallet snapshot map
    plan      Phase 8  clearance-safe routes (forklift->pallet, pallet->drop-off)

World frame (CLAUDE.md): right-handed metres, origin = centre of floor,
+X along the 51.73 m length, +Y across the 26.3 m depth, +Z up, floor = Z=0.
"""

__version__ = "0.1.0"
