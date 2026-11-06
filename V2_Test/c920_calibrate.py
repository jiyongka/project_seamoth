"""
c920_calibrate.py -- calibrate the Logitech C920 with a printed checkerboard.
Saves every camera parameter to c920_calib.npz, which c920_tag.py loads.

Usage:
    python c920_calibrate.py capture    # SPACE = save a view, Q = quit
    python c920_calibrate.py solve      # compute + save c920_calib.npz
No distances need to be measured. Only the square size must be (ruler/calipers).

Rules: keep the SAME resolution and focus later in c920_tag.py (they are stored in the file).
Recalibrate if you change resolution, focus, zoom, or go underwater.
"""
import glob
import os
import sys

import cv2 as cv
import numpy as np

# ------------------------------------------------------------ EDIT THESE ----
CAMERA_INDEX = 0
WIDTH, HEIGHT = 1280, 720          # calibrate at the resolution you will use
INNER_COLS, INNER_ROWS = 9, 6      # INNER corners (squares - 1): 10x7 squares -> 9x6
SQUARE_SIZE_M = 0.025              # MEASURE one printed square, in metres
AUTOFOCUS = 0                      # 0 = off (locked), 1 = on
FOCUS = 30                         # C920 focus 0-250 (higher = farther). Ignored if not supported
MIN_IMAGES = 15
IMAGE_DIR = "calib_images"
OUT_FILE = "c920_calib.npz"
# -----------------------------------------------------------------------------

PATTERN = (INNER_COLS, INNER_ROWS)
CRIT = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def open_camera():
    cap = cv.VideoCapture(CAMERA_INDEX)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv.CAP_PROP_AUTOFOCUS, AUTOFOCUS)
    if not AUTOFOCUS:
        cap.set(cv.CAP_PROP_FOCUS, FOCUS)
    if not cap.isOpened():
        sys.exit("camera did not open (check permission / CAMERA_INDEX)")
    for _ in range(10):          # let exposure settle
        cap.read()
    w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f"camera running at {w}x{h}; autofocus={cap.get(cv.CAP_PROP_AUTOFOCUS)} "
          f"focus={cap.get(cv.CAP_PROP_FOCUS)}")
    if (w, h) != (WIDTH, HEIGHT):
        print(f"WARNING: asked for {WIDTH}x{HEIGHT}, got {w}x{h}. Calibration will be for {w}x{h}.")
    return cap


def find_corners(gray):
    ok, c = cv.findChessboardCorners(
        gray, PATTERN, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return False, None
    return True, cv.cornerSubPix(gray, c, (11, 11), (-1, -1), CRIT)


def capture():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    cap = open_camera()
    n = len(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    print("Move the board: near/far, tilted up to ~40 deg, and into every corner of the frame.")
    print("SPACE = save (only when detected), Q = quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        found, corners = find_corners(gray)
        view = frame.copy()
        if found:
            cv.drawChessboardCorners(view, PATTERN, corners, found)
        msg = f"saved {n} (need {MIN_IMAGES}+)   board {'DETECTED' if found else 'not found'}"
        cv.putText(view, msg, (15, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                   (0, 255, 0) if found else (0, 0, 255), 2, cv.LINE_AA)
        cv.imshow("c920 calibration capture", view)
        k = cv.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == 32 and found:
            cv.imwrite(os.path.join(IMAGE_DIR, f"calib_{n:03d}.png"), frame)  # raw frame
            n += 1
            print("saved", n)
    cap.release()
    cv.destroyAllWindows()


def solve():
    paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    if not paths:
        sys.exit(f"no images in {IMAGE_DIR}/ -- run: python c920_calibrate.py capture")
    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_SIZE_M
    objpts, imgpts, used, size = [], [], [], None
    for p in paths:
        g = cv.imread(p, cv.IMREAD_GRAYSCALE)
        if size is None:
            size = g.shape[::-1]
        if g.shape[::-1] != size:
            print("skip (different resolution):", p)
            continue
        ok, c = find_corners(g)
        if ok:
            objpts.append(objp)
            imgpts.append(c)
            used.append(p)
        else:
            print("skip (board not found):", p)
    if len(used) < MIN_IMAGES:
        sys.exit(f"only {len(used)} usable images, need {MIN_IMAGES}+")

    rms, K, dist, rvecs, tvecs = cv.calibrateCamera(objpts, imgpts, size, None, None)

    errs = []
    for i in range(len(used)):
        proj, _ = cv.projectPoints(objpts[i], rvecs[i], tvecs[i], K, dist)
        d = imgpts[i].reshape(-1, 2).astype(np.float64) - proj.reshape(-1, 2).astype(np.float64)
        errs.append(float(np.sqrt((d ** 2).sum(axis=1).mean())))

    np.savez(OUT_FILE, K=K, dist=dist, width=size[0], height=size[1], rms=rms,
             square_size_m=SQUARE_SIZE_M, autofocus=AUTOFOCUS, focus=FOCUS)

    print(f"\nused {len(used)}/{len(paths)} images at {size[0]}x{size[1]}")
    print(f"RMS reprojection error: {rms:.3f} px   (under ~0.5 is good)")
    print(f"fx {K[0,0]:.1f}  fy {K[1,1]:.1f}  cx {K[0,2]:.1f}  cy {K[1,2]:.1f}")
    print("distortion k1 k2 p1 p2 k3:", np.round(dist.ravel(), 5))
    f_spec = np.hypot(*size) / 2 / np.tan(np.radians(78 / 2))
    print(f"datasheet check: 78 deg diagonal FOV -> f ~ {f_spec:.0f} px (yours {K[0,0]:.0f}); "
          "within ~5-10% is normal")
    print("worst images (delete any far above the rest, then re-run solve):")
    for i in np.argsort(errs)[::-1][:5]:
        print(f"   {used[i]}  {errs[i]:.3f} px")
    print(f"\nsaved {OUT_FILE}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "capture":
        capture()
    elif cmd == "solve":
        solve()
    else:
        print(__doc__)
