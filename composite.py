#!/usr/bin/env python3
"""Composite an AI-restyled video inside a tracked finger frame.

Tracks the finger-frame gesture (both hands, index + thumb "L"s) in the
original footage with MediaPipe Hand Landmarker, then reveals the stylized
video through the quad the fingers form — the same window effect as the
finger-frame-effect web app, with the dashed outline and corner dots.

The tracking logic is a direct port of the web app's audited pipeline:
anatomical corner ordering (crossed fingers = bowtie), spread/area gates
with hysteresis, teleport rejection, velocity-adaptive smoothing, dropout
hold, and presence fade.

Usage:
    python composite.py finger-effect-raw.mov stylized.mp4 -o final.mp4
"""

import argparse
import math
import os
import subprocess
import sys
import urllib.request

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = "hand_landmarker.task"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_MODEL_PATH = "face_landmarker.task"
# Outer eye corners in the MediaPipe face mesh.
EYE_R, EYE_L = 33, 263

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9

# Tracking constants — mirror main.js in the web app.
MAX_LOST_FRAMES = 25
JUMP_CONFIRM_FRAMES = 2
JUMP_FRACTION = 0.3
ALPHA_MIN, ALPHA_MAX = 0.35, 0.85
ALPHA_SCALE = 0.05
PRESENCE_IN, PRESENCE_OUT = 0.12, 0.05
SPREAD_ACQUIRE, SPREAD_KEEP = 0.75, 0.2
AREA_ACQUIRE, AREA_KEEP = 0.005, 0.0005


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def lerp_pt(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def polygon_area(pts):
    a = 0.0
    for i in range(len(pts)):
        p, q = pts[i], pts[(i + 1) % len(pts)]
        a += p[0] * q[1] - q[0] * p[1]
    return abs(a / 2)


def angle_sorted(pts):
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


class FrameTracker:
    """Stateful quad tracker, ported from the web app's main loop."""

    def __init__(self, width, height):
        self.w, self.h = width, height
        self.corners = None
        self.presence = 0.0
        self.frame_active = False
        self.lost_frames = 0
        self.jump_frames = 0

    def compute_quad(self, hands):
        """hands: list of landmark lists (normalized). Returns anatomical quad
        [A.index, B.index, B.thumb, A.thumb] (A = smaller wrist x) or None."""
        if len(hands) != 2:
            return None
        info = []
        for lm in hands:
            px = lambda i: (lm[i].x * self.w, lm[i].y * self.h)
            index, thumb = px(INDEX_TIP), px(THUMB_TIP)
            scale = dist(px(WRIST), px(MIDDLE_MCP)) + 1
            needed = SPREAD_KEEP if self.frame_active else SPREAD_ACQUIRE
            if dist(thumb, index) < scale * needed:
                return None
            info.append({"index": index, "thumb": thumb, "wx": px(WRIST)[0]})
        info.sort(key=lambda hd: hd["wx"])
        a, b = info
        pts = [a["index"], b["index"], b["thumb"], a["thumb"]]
        min_area = AREA_KEEP if self.frame_active else AREA_ACQUIRE
        if polygon_area(angle_sorted(pts)) < self.w * self.h * min_area:
            return None
        return pts

    def update(self, hands):
        target = self.compute_quad(hands) if hands else None

        if target:
            if self.corners is None:
                self.lost_frames = 0
                self.frame_active = True
                self.jump_frames = 0
                self.corners = target
                self.presence = min(1.0, self.presence + PRESENCE_IN)
            else:
                moved = sum(dist(p, c) for p, c in zip(target, self.corners)) / 4
                if (
                    moved > self.w * JUMP_FRACTION
                    and self.jump_frames + 1 < JUMP_CONFIRM_FRAMES
                ):
                    self.jump_frames += 1
                    self.lost_frames += 1
                    if self.lost_frames > MAX_LOST_FRAMES:
                        self.presence = max(0.0, self.presence - PRESENCE_OUT)
                else:
                    self.lost_frames = 0
                    self.frame_active = True
                    self.jump_frames = 0
                    alpha = min(
                        ALPHA_MAX, max(ALPHA_MIN, moved / (self.w * ALPHA_SCALE))
                    )
                    self.corners = [
                        lerp_pt(c, p, alpha) for c, p in zip(self.corners, target)
                    ]
                    self.presence = min(1.0, self.presence + PRESENCE_IN)
        elif self.corners is not None and self.lost_frames < MAX_LOST_FRAMES:
            self.lost_frames += 1
            self.presence = min(1.0, self.presence + PRESENCE_IN)
        else:
            self.presence = max(0.0, self.presence - PRESENCE_OUT)
            if self.presence == 0:
                self.corners = None
                self.frame_active = False
                self.jump_frames = 0

        return self.corners if self.presence > 0.01 else None


def draw_outline(frame, quad, presence, t):
    """Dashed marching-ants outline + pulsing corner dots, like the web app."""
    overlay = frame.copy()

    # Dashed edges with a marching offset.
    dash_on, dash_off = 10.0, 8.0
    period = dash_on + dash_off
    offset = (t * 40.0) % period
    for i in range(4):
        p0 = np.array(quad[i], dtype=float)
        p1 = np.array(quad[(i + 1) % 4], dtype=float)
        seg = p1 - p0
        length = float(np.hypot(*seg))
        if length < 1:
            continue
        u = seg / length
        s = -offset
        while s < length:
            a = max(0.0, s)
            b = min(length, s + dash_on)
            if b > a:
                pa = (p0 + u * a).astype(int)
                pb = (p0 + u * b).astype(int)
                cv2.line(overlay, tuple(pa), tuple(pb), (242, 242, 242), 2, cv2.LINE_AA)
            s += period

    # Corner dots with pulse + expanding halo.
    for i, p in enumerate(quad):
        c = (int(p[0]), int(p[1]))
        r = 7 + math.sin(t * 3 + i * 1.5) * 1.5
        halo = (t * 0.8 + i * 0.25) % 1.0
        halo_val = int(255 * 0.5 * (1 - halo))
        cv2.circle(overlay, c, int(r + halo * 14), (halo_val,) * 3, 2, cv2.LINE_AA)
        cv2.circle(overlay, c, int(r), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, c, int(r), (60, 60, 60), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, presence, frame, 1 - presence, 0, dst=frame)


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model …")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    if not os.path.exists(FACE_MODEL_PATH):
        print("Downloading face landmarker model …")
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)


class FaceAligner:
    """Estimate a smoothed similarity transform (scale + translation) that
    maps the stylized video's face onto the original's, correcting AI models
    that redraw the person slightly smaller/larger or shifted."""

    def __init__(self, vision_mod, mp_tasks_mod, mp_mod, w, h):
        self.mp = mp_mod
        self.w, self.h = w, h

        def opts():
            return vision_mod.FaceLandmarkerOptions(
                base_options=mp_tasks_mod.BaseOptions(
                    model_asset_path=FACE_MODEL_PATH
                ),
                running_mode=vision_mod.RunningMode.VIDEO,
                num_faces=1,
            )

        self.det_orig = vision_mod.FaceLandmarker.create_from_options(opts())
        self.det_sty = vision_mod.FaceLandmarker.create_from_options(opts())
        self.s, self.tx, self.ty = 1.0, 0.0, 0.0
        self.have = False

    def _eyes(self, det, frame_bgr, ts):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = det.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), ts
        )
        if not res.face_landmarks:
            return None
        lm = res.face_landmarks[0]
        p = lambda i: (lm[i].x * self.w, lm[i].y * self.h)
        r, l = p(EYE_R), p(EYE_L)
        mid = ((r[0] + l[0]) / 2, (r[1] + l[1]) / 2)
        return mid, dist(r, l)

    def update(self, orig_frame, sty_frame, ts):
        eo = self._eyes(self.det_orig, orig_frame, ts)
        es = self._eyes(self.det_sty, sty_frame, ts)
        if eo and es and es[1] > 1:
            s = eo[1] / es[1]
            tx = eo[0][0] - s * es[0][0]
            ty = eo[0][1] - s * es[0][1]
            # EMA smoothing so the correction doesn't jitter.
            a = 0.15 if self.have else 1.0
            self.s += (s - self.s) * a
            self.tx += (tx - self.tx) * a
            self.ty += (ty - self.ty) * a
            self.have = True

    def apply(self, sty_frame):
        if not self.have or abs(self.s - 1) + abs(self.tx) + abs(self.ty) < 1e-3:
            return sty_frame
        m = np.float32([[self.s, 0, self.tx], [0, self.s, self.ty]])
        return cv2.warpAffine(
            sty_frame, m, (self.w, self.h), borderMode=cv2.BORDER_REPLICATE
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("original", nargs="?", default="finger-effect-raw.mov")
    ap.add_argument("stylized", nargs="?", default="stylized.mp4")
    ap.add_argument("-o", "--output", default="final.mp4")
    ap.add_argument(
        "--no-align",
        action="store_true",
        help="disable face-based auto-alignment of the stylized video",
    )
    args = ap.parse_args()

    for f in (args.original, args.stylized):
        if not os.path.exists(f):
            sys.exit(f"Missing input: {f}")
    ensure_model()

    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    cap = cv2.VideoCapture(args.original)
    sty = cv2.VideoCapture(args.stylized)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sty_fps = sty.get(cv2.CAP_PROP_FPS) or fps
    sty_count = int(sty.get(cv2.CAP_PROP_FRAME_COUNT))

    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    )
    aligner = None if args.no_align else FaceAligner(vision, mp_tasks, mp, w, h)

    # Pipe frames straight into ffmpeg for a proper H.264 output.
    ff = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
            "-r", f"{fps}", "-i", "-",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            args.output,
        ],
        stdin=subprocess.PIPE,
    )

    tracker = FrameTracker(w, h)
    sty_frames = []
    i = 0
    tracked = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Cache stylized frames lazily (they're reused when fps differ).
        t = i / fps
        j = min(int(round(t * sty_fps)), max(sty_count - 1, 0))
        while len(sty_frames) <= j:
            ok_s, sf = sty.read()
            if not ok_s:
                break
            if sf.shape[:2] != (h, w):
                sf = cv2.resize(sf, (w, h))
            sty_frames.append(sf)
        sty_frame = sty_frames[min(j, len(sty_frames) - 1)] if sty_frames else None

        ts = int(i * 1000 / fps)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
        )
        quad = tracker.update(result.hand_landmarks or [])

        if aligner is not None and sty_frame is not None:
            aligner.update(frame, sty_frame, ts)
            sty_frame = aligner.apply(sty_frame)

        if quad is not None and sty_frame is not None:
            tracked += 1
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(quad, dtype=np.int32)], 255)
            m = (mask.astype(np.float32) / 255.0 * tracker.presence)[..., None]
            frame = (
                frame.astype(np.float32) * (1 - m)
                + sty_frame.astype(np.float32) * m
            ).astype(np.uint8)
            draw_outline(frame, quad, tracker.presence, t)

        ff.stdin.write(frame.tobytes())
        i += 1
        if i % 30 == 0:
            print(f"  frame {i}, frame visible on {tracked} frames so far")

    ff.stdin.close()
    ff.wait()
    cap.release()
    sty.release()

    # Carry over the original audio track if there is one.
    has_audio = (
        subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=codec_type", "-of", "csv=p=0", args.original],
            capture_output=True, text=True,
        ).stdout.strip() != ""
    )
    if has_audio:
        tmp = args.output + ".mux.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", args.output,
             "-i", args.original, "-map", "0:v", "-map", "1:a",
             "-c:v", "copy", "-c:a", "aac", "-shortest", tmp],
            check=True,
        )
        os.replace(tmp, args.output)

    print(f"Done: {args.output} ({i} frames, finger frame visible on {tracked})")


if __name__ == "__main__":
    main()
