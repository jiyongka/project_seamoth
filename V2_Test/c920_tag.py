"""
c920_tag.py -- AprilTag distance + position with the C920, using c920_calib.npz.
Same detection and overlay as mac_tag.py, but uses the real calibration
(focal lengths, optical center, lens distortion) instead of a guessed field of view.

Usage:
    python c920_tag.py --size 0.10
    python c920_tag.py --size 0.10 --families tag25h9
Keys: ESC/q quit   SPACE print one reading to the terminal
"""
import argparse
import os
import sys
import time

import cv2 as cv
import numpy as np
from pupil_apriltags import Detector

ap = argparse.ArgumentParser()
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--calib", default="c920_calib.npz")
ap.add_argument("--families", default="tag36h11")            # tag36h11 / tag25h9 / tag16h5
ap.add_argument("--size", type=float, default=0.10, help="black square side, metres (MEASURE IT)")
ap.add_argument("--quad_decimate", type=float, default=1.0, help="1.0 = most accurate, 2.0 = faster")
ap.add_argument("--image", help="run on a still image instead of the camera")
a = ap.parse_args()

if not os.path.exists(a.calib):
    sys.exit(f"{a.calib} not found. Run c920_calibrate.py (capture, then solve) first.")
cal = np.load(a.calib)
K, DIST = cal["K"], cal["dist"]
W, H = int(cal["width"]), int(cal["height"])
print(f"loaded {a.calib}: {W}x{H}  fx {K[0,0]:.0f}  fy {K[1,1]:.0f}  rms {float(cal['rms']):.2f}px")

det = Detector(families=a.families, nthreads=1, quad_decimate=a.quad_decimate,
               quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)

# Undistort once with a remap table; the new camera matrix is K itself, so the pose maths
# can use K directly afterwards.
MAP1, MAP2 = cv.initUndistortRectifyMap(K, DIST, None, K, (W, H), cv.CV_16SC2)


def process(frame):
    h, w = frame.shape[:2]
    if (w, h) != (W, H):
        sys.exit(f"frame is {w}x{h} but calibration is for {W}x{H}. Re-run c920_calibrate.py "
                 "at the new resolution (or change the camera resolution).")
    und = cv.remap(frame, MAP1, MAP2, cv.INTER_LINEAR)
    gray = cv.cvtColor(und, cv.COLOR_BGR2GRAY)
    tags = det.detect(gray, estimate_tag_pose=True,
                      camera_params=[K[0, 0], K[1, 1], K[0, 2], K[1, 2]], tag_size=a.size)
    out = und.copy()
    results, texts = [], [(f"{a.families}  size {a.size * 100:.1f} cm  calibrated  {w}x{h}  "
                           f"tags {len(tags)}", (0, 255, 0))]
    for t in tags:
        R, tv = np.asarray(t.pose_R), np.asarray(t.pose_t).reshape(3)
        cam_in_tag = -R.T @ tv                     # camera position in the tag's frame
        rng = float(np.linalg.norm(tv))
        tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        px = float(np.linalg.norm(t.corners[0] - t.corners[1]))
        results.append(dict(id=t.tag_id, t=tv, cam=cam_in_tag, range=rng, tilt=tilt, px=px))

        cv.polylines(out, [t.corners.astype(int)], True, (0, 255, 0), 2)
        cv.circle(out, (int(t.center[0]), int(t.center[1])), 5, (0, 0, 255), 2)
        rvec, _ = cv.Rodrigues(R)
        cv.drawFrameAxes(out, K, np.zeros(5), rvec, tv.reshape(3, 1), a.size * 0.6)
        texts += [(s, (0, 255, 0)) for s in (
            f"id {t.tag_id}   RANGE {rng * 100:.1f} cm   tilt {tilt:.0f} deg",
            f"tag in cam  x {tv[0] * 100:+.1f}  y {tv[1] * 100:+.1f}  z {tv[2] * 100:+.1f} cm",
            f"cam in tag  x {cam_in_tag[0] * 100:+.1f}  y {cam_in_tag[1] * 100:+.1f}  "
            f"z {cam_in_tag[2] * 100:+.1f} cm",
            f"{px:.0f}px wide")]
    for i, (s, col) in enumerate(texts):          # readings in the top-left corner
        cv.putText(out, s, (10, 25 + 24 * i), cv.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv.LINE_AA)
    return out, results


if a.image:
    img = cv.imread(a.image)
    _, res = process(img)
    for r in res:
        print(f"id {r['id']}  range {r['range'] * 100:.1f} cm  tag-in-cam {np.round(r['t'] * 100, 1)} "
              f"cam-in-tag {np.round(r['cam'] * 100, 1)}  tilt {r['tilt']:.0f}")
    raise SystemExit

cap = cv.VideoCapture(a.device)
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
cap.set(cv.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv.CAP_PROP_AUTOFOCUS, int(cal["autofocus"]))      # same focus as during calibration
if not int(cal["autofocus"]):
    cap.set(cv.CAP_PROP_FOCUS, float(cal["focus"]))
if not cap.isOpened():
    raise SystemExit("camera did not open (check camera permission / --device)")

last = []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    view, last = process(frame)
    cv.imshow("C920 AprilTag", view)
    k = cv.waitKey(1) & 0xFF
    if k in (27, ord("q")):
        break
    if k == 32 and last:
        r = last[0]
        print(f"{time.strftime('%H:%M:%S')} id {r['id']} range {r['range'] * 100:.1f} cm "
              f"tag-in-cam {np.round(r['t'] * 100, 1)} cam-in-tag {np.round(r['cam'] * 100, 1)}")
cap.release()
cv.destroyAllWindows()
