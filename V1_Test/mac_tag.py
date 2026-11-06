"""
mac_tag.py -- AprilTag detection + distance/position overlay for the MacBook camera.
Detection settings copied from the working reference demo (grayscale -> Detector.detect),
with pose estimation turned on.

Usage:
    python mac_tag.py --size 0.10                 # size = black square side in METRES
    python mac_tag.py --size 0.10 --ref 0.50      # then hold tag at 0.50 m and press C
Keys:  ESC/q quit    c  set focal length from --ref distance    SPACE print a line
"""
import argparse
import time

import cv2 as cv
import numpy as np
from pupil_apriltags import Detector

ap = argparse.ArgumentParser()
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--width", type=int, default=960)
ap.add_argument("--height", type=int, default=540)
ap.add_argument("--families", default="tag36h11")           # tag36h11 / tag25h9 / tag16h5
ap.add_argument("--size", type=float, default=0.10, help="black square side, metres (MEASURE IT)")
ap.add_argument("--fov", type=float, default=70.0, help="assumed DIAGONAL fov, degrees (a guess)")
ap.add_argument("--ref", type=float, default=None, help="known distance (m) for the C key")
ap.add_argument("--quad_decimate", type=float, default=2.0)
ap.add_argument("--image", help="run on a still image instead of the camera")
a = ap.parse_args()

det = Detector(families=a.families, nthreads=1, quad_decimate=a.quad_decimate,
               quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)

f_override = None          # set by the C key
PANEL, STATUS_COLOR, VALUE_COLOR = (225, 225, 225), (0, 90, 0), (0, 0, 0)   # BGR


def focal(w, h):
    if f_override:
        return f_override
    return np.hypot(w, h) / 2 / np.tan(np.radians(a.fov / 2))


def process(frame):
    """Returns (annotated frame, list of result dicts)."""
    h, w = frame.shape[:2]
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    f = focal(w, h)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    tags = det.detect(gray, estimate_tag_pose=True, camera_params=[f, f, w / 2, h / 2],
                      tag_size=a.size)
    out = frame.copy()
    results = []
    hud = []                                           # (text, colour) lines for the top-left corner
    for t in tags:
        R, tv = np.asarray(t.pose_R), np.asarray(t.pose_t).reshape(3)
        cam_in_tag = -R.T @ tv                         # camera position in the tag's frame
        rng = float(np.linalg.norm(tv))
        tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        px = float(np.linalg.norm(t.corners[0] - t.corners[1]))
        results.append(dict(id=t.tag_id, t=tv, cam=cam_in_tag, range=rng, tilt=tilt, px=px))

        c = t.corners.astype(int)
        cv.polylines(out, [c], True, (0, 255, 0), 2)
        cv.circle(out, (int(t.center[0]), int(t.center[1])), 5, (0, 0, 255), 2)
        rvec, _ = cv.Rodrigues(R)
        cv.drawFrameAxes(out, K, np.zeros(5), rvec, tv.reshape(3, 1), a.size * 0.6)
        hud.append((f"id {t.tag_id}   RANGE {rng * 100:.1f} cm   tilt {tilt:.0f} deg   {px:.0f}px wide", VALUE_COLOR))
        hud.append((f"tag in cam  x {tv[0] * 100:+.1f}  y {tv[1] * 100:+.1f}  z {tv[2] * 100:+.1f} cm", VALUE_COLOR))
        hud.append((f"cam in tag  x {cam_in_tag[0] * 100:+.1f}  y {cam_in_tag[1] * 100:+.1f}  "
                    f"z {cam_in_tag[2] * 100:+.1f} cm", VALUE_COLOR))
    status = (f"{a.families}  size {a.size * 100:.1f} cm  f={f:.0f}px "
              f"({'set by C key' if f_override else f'from fov {a.fov:g}'})  {w}x{h}  tags {len(tags)}")
    hud.insert(0, (status, STATUS_COLOR))
    panel_w = max(cv.getTextSize(s, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] for s, _ in hud)
    cv.rectangle(out, (0, 0), (panel_w + 20, 24 * len(hud) + 12), PANEL, cv.FILLED)
    for i, (s, color) in enumerate(hud):
        cv.putText(out, s, (10, 25 + 24 * i), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv.LINE_AA)
    return out, results


def set_focal_from_ref(frame, results):
    """If the tag is held at a=--ref metres, f scales so the measured range equals it."""
    global f_override
    if a.ref is None or not results:
        print("press C with --ref set and a tag in view")
        return
    r = max(results, key=lambda d: d["px"])
    f_override = focal(*frame.shape[1::-1]) * a.ref / r["range"]
    print(f"focal length set to {f_override:.0f} px using ref {a.ref} m")


if a.image:
    img = cv.imread(a.image)
    _, res = process(img)
    for r in res:
        print(f"id {r['id']}  range {r['range'] * 100:.1f} cm  tag-in-cam {np.round(r['t'] * 100, 1)} "
              f"cam-in-tag {np.round(r['cam'] * 100, 1)}  tilt {r['tilt']:.0f}")
    raise SystemExit

cap = cv.VideoCapture(a.device)
cap.set(cv.CAP_PROP_FRAME_WIDTH, a.width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, a.height)
if not cap.isOpened():
    raise SystemExit("camera did not open (check camera permission / --device)")

last_res = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    view, last_res = process(frame)
    cv.imshow("AprilTag distance", view)
    k = cv.waitKey(1) & 0xFF
    if k in (27, ord("q")):
        break
    if k == ord("c"):
        set_focal_from_ref(frame, last_res)
    if k == 32 and last_res:
        r = last_res[0]
        print(f"{time.strftime('%H:%M:%S')} id {r['id']} range {r['range'] * 100:.1f} cm "
              f"tag-in-cam {np.round(r['t'] * 100, 1)} cam-in-tag {np.round(r['cam'] * 100, 1)}")
cap.release()
cv.destroyAllWindows()