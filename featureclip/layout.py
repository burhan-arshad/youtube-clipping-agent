"""Find the *gameplay* inside a stream frame and choose a vertical crop that
contains only gameplay (no streamer facecam, no stream overlay, no borders).

Two steps:
  1. detect_layout(): once per source video
       - rows/columns that never change over time = static overlay/border -> cut away
       - facecam (a face that stays in the same place) -> a box the crop must avoid
  2. choose_crop(): once per clip
       - slide a 9:16 (or 4:5) window over the gameplay rectangle, pick the position
         with the most motion/detail, never overlapping the facecam box.
If auto-detection is wrong for a streamer, put a manual `crop: [x, y, w, h]` in sources.yaml.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x, y, w, h


@dataclass
class Layout:
    frame_w: int
    frame_h: int
    rect: Box
    avoid: list[Box] = field(default_factory=list)
    origin: str = "auto"  # auto | manual | full
    cam: Box | None = None   # the streamer's webcam (used by the "streamer" layout: webcam on top, gameplay below)

    def to_dict(self) -> dict:
        return {"frame": [self.frame_w, self.frame_h], "rect": list(self.rect),
                "avoid": [list(b) for b in self.avoid], "origin": self.origin,
                "cam": list(self.cam) if self.cam else None}

    @classmethod
    def from_dict(cls, d: dict) -> "Layout":
        cam = d.get("cam")
        return cls(d["frame"][0], d["frame"][1], tuple(d["rect"]), [tuple(b) for b in d.get("avoid", [])],
                   d.get("origin", "auto"), tuple(cam) if cam else None)


# ------------------------------------------------------------------ frame sampling

def sample_frames(video: str | Path, times: list[float], width: int = 640) -> list[np.ndarray]:
    """BGR frames at the given times (fast keyframe seek, one ffmpeg call per frame)."""
    frames: list[np.ndarray] = []
    for t in times:
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", str(video),
               "-frames:v", "1", "-vf", f"scale={width}:-2", "-f", "image2pipe",
               "-vcodec", "bmp", "-"]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if not p.stdout:
            continue
        img = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    return frames


# ------------------------------------------------------------------ geometry helpers

def _inter(a: Box, b: Box) -> bool:
    return not (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])


def _even(v: float) -> int:
    return int(v) // 2 * 2


def _exclude(rects: list[Box], box: Box, min_side: int = 160) -> list[Box]:
    """All sub-rectangles of each rect that do not touch `box`."""
    out: list[Box] = []
    for r in rects:
        if not _inter(r, box):
            out.append(r)
            continue
        rx, ry, rw, rh = r
        bx, by, bw, bh = box
        options = [
            (rx, ry, max(0, bx - rx), rh),                         # left of box
            (bx + bw, ry, max(0, rx + rw - (bx + bw)), rh),        # right of box
            (rx, ry, rw, max(0, by - ry)),                         # above box
            (rx, by + bh, rw, max(0, ry + rh - (by + bh))),        # below box
        ]
        out += [o for o in options if o[2] >= min_side and o[3] >= min_side]
    return out


def cluster_faces(per_frame: list[list[Box]], n_frames: int, min_presence: float = 0.25,
                  radius: float = 0.06, frame_w: int = 640) -> list[Box]:
    """Group face boxes that stay at the same place; keep groups seen in >= min_presence of frames."""
    groups: list[dict] = []
    tol = radius * frame_w
    for boxes in per_frame:
        for (x, y, w, h) in boxes:
            cx, cy = x + w / 2, y + h / 2
            for g in groups:
                if abs(g["cx"] - cx) <= tol and abs(g["cy"] - cy) <= tol:
                    g["n"] += 1
                    g["cx"] = (g["cx"] * (g["n"] - 1) + cx) / g["n"]
                    g["cy"] = (g["cy"] * (g["n"] - 1) + cy) / g["n"]
                    g["w"] = (g["w"] * (g["n"] - 1) + w) / g["n"]
                    g["h"] = (g["h"] * (g["n"] - 1) + h) / g["n"]
                    break
            else:
                groups.append({"cx": cx, "cy": cy, "w": w, "h": h, "n": 1})
    keep = sorted((g for g in groups if g["n"] >= max(2, min_presence * n_frames)), key=lambda g: -g["n"])
    return [(int(g["cx"] - g["w"] / 2), int(g["cy"] - g["h"] / 2), int(g["w"]), int(g["h"])) for g in keep]


def facecam_box(face: Box, fw: int, fh: int, grow: tuple[float, float] = (2.4, 2.0),
                edge_snap: float = 0.18) -> Box:
    """A webcam window is bigger than the face itself and usually touches a screen edge."""
    x, y, w, h = face
    cx, cy = x + w / 2, y + h / 2
    bw, bh = w * grow[0], h * grow[1]
    x0, y0 = max(0, cx - bw / 2), max(0, cy - bh / 2)
    x1, y1 = min(fw, cx + bw / 2), min(fh, cy + bh / 2)
    if x0 < edge_snap * fw:
        x0 = 0
    if fw - x1 < edge_snap * fw:
        x1 = fw
    if y0 < edge_snap * fh:
        y0 = 0
    if fh - y1 < edge_snap * fh:
        y1 = fh
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


# ------------------------------------------------------------------ step 1: layout of the source

DATA = Path(__file__).resolve().parent / "data"
CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")


def corner_box(corner: str, fw: int, fh: int, size: tuple[float, float] = (0.27, 0.38)) -> Box:
    """A fixed facecam area in one corner of the frame (width/height as fractions of the frame)."""
    w, h = _even(fw * size[0]), _even(fh * size[1])
    x = 0 if "left" in corner else fw - w
    y = 0 if "top" in corner else fh - h
    return (x, y, w, h)


def find_faces(gray_frames: list[np.ndarray], frame_w: int) -> list[Box]:
    """Faces that stay at about the same place (= a facecam). Uses 3 detectors, so a streamer who
    turns the head or leans back is still found. The cascade files ship with the project."""
    dets = []
    for name in ("haarcascade_frontalface_alt2.xml", "haarcascade_frontalface_default.xml",
                 "haarcascade_profileface.xml"):
        for path in (DATA / name, Path(getattr(cv2.data, "haarcascades", "")) / name):
            if path.is_file():
                d = cv2.CascadeClassifier(str(path))
                if not d.empty():
                    dets.append(d)
                    break
    if not dets:
        print("WARNING: no face detector files found, facecam cannot be detected automatically. "
              "Set layout: {facecam: top-right} in config.yaml.")
        return []
    min_size = max(24, int(frame_w * 0.04))
    per_frame = []
    for g in gray_frames:
        found: list[Box] = []
        for d in dets:
            for f in d.detectMultiScale(g, scaleFactor=1.1, minNeighbors=4, minSize=(min_size, min_size)):
                found.append(tuple(int(v) for v in f))
        per_frame.append(found)
    fh, fw = gray_frames[0].shape[:2]
    for (x, y, w, h) in cluster_faces(per_frame, len(gray_frames), min_presence=0.08, frame_w=frame_w):
        cx, cy = x + w / 2, y + h / 2
        if not ((cx < 0.3 * fw or cx > 0.7 * fw) and (cy < 0.4 * fh or cy > 0.6 * fh)):
            continue                      # a webcam sits in a corner; faces in the middle are the game
        box = facecam_box((x, y, w, h), fw, fh)
        if box[2] * box[3] > 0.3 * fw * fh:
            continue
        return [box]                      # one streamer = one facecam (clusters come most-seen first)
    return []


def _covered(b: Box, L: Layout) -> bool:
    cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
    rx, ry, rw, rh = L.rect
    if not (rx <= cx <= rx + rw and ry <= cy <= ry + rh):
        return True                                  # outside the gameplay area: already cut away
    return any(a[0] <= cx <= a[0] + a[2] and a[1] <= cy <= a[1] + a[3] for a in L.avoid)


def _add_missed_webcam(L: Layout, video, duration: float) -> Layout:
    """Safety net: if a real webcam face sits inside the gameplay area but the AI did not box it, add it."""
    try:
        times = list(np.linspace(duration * 0.1, duration * 0.9, 10)) if duration > 10 else [duration / 2]
        frames = sample_frames(video, times, 640)
        if len(frames) < 4:
            return L
        sh, sw = frames[0].shape[:2]
        sx, sy = L.frame_w / sw, L.frame_h / sh
        for b in find_faces([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames], sw):
            full = (_even(b[0] * sx), _even(b[1] * sy), _even(b[2] * sx), _even(b[3] * sy))
            if not _covered(full, L):
                print(f"[layout] a webcam the AI missed was added: {full}")
                L.avoid.append(full)
    except Exception:
        pass
    return L


def detect_layout(video: str | Path, duration: float, frame_w: int, frame_h: int,
                  n_frames: int = 24, face_detect: bool = True,
                  manual: list[int] | None = None, facecam: str = "auto",
                  facecam_size: tuple[float, float] = (0.27, 0.38),
                  crop_frac: list[float] | None = None, llm=None, tmp: Path | None = None) -> Layout:
    if llm is not None and not manual and not crop_frac and (facecam or "auto").lower() == "auto":
        from . import ai_layout
        L = ai_layout.detect(llm, video, duration, frame_w, frame_h, Path(tmp or Path(video).parent / "layout_tmp"))
        if L is not None:
            return _add_missed_webcam(L, video, duration)
    if crop_frac and not manual:     # gameplay rectangle as fractions of the frame: [x, y, w, h]
        manual = [crop_frac[0] * frame_w, crop_frac[1] * frame_h, crop_frac[2] * frame_w, crop_frac[3] * frame_h]
    if manual:
        x, y, w, h = [int(v) for v in manual]
        return Layout(frame_w, frame_h, (_even(x), _even(y), _even(w), _even(h)), [], "manual")

    lo, hi = duration * 0.05, duration * 0.95
    times = list(np.linspace(lo, hi, n_frames)) if duration > 10 else [duration / 2]
    frames = sample_frames(video, times, 640)
    if len(frames) < 4:
        return Layout(frame_w, frame_h, (0, 0, _even(frame_w), _even(frame_h)), [], "full")
    sh, sw = frames[0].shape[:2]
    sx, sy = frame_w / sw, frame_h / sh

    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)
    std = gray.std(axis=0)
    active = std > 3.0
    row_frac = active.mean(axis=1)
    col_frac = active.mean(axis=0)
    rows = np.where(row_frac >= 0.15)[0]
    cols = np.where(col_frac >= 0.15)[0]
    if len(rows) and len(cols):
        x0, x1, y0, y1 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
    else:
        x0, y0, x1, y1 = 0, 0, sw, sh
    rect_small = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
    if rect_small[2] * rect_small[3] < 0.25 * sw * sh:
        rect_small = (0, 0, sw, sh)
        origin = "full"
    else:
        origin = "auto"
    # trim a couple of pixels so compression halos around overlays are not in the crop
    pad = 3 if origin == "auto" and rect_small != (0, 0, sw, sh) else 0
    rect_small = (rect_small[0] + pad, rect_small[1] + pad,
                  rect_small[2] - 2 * pad, rect_small[3] - 2 * pad)

    avoid: list[Box] = []
    facecam = (facecam or "auto").lower()
    if facecam in CORNERS:                       # the user told us where the webcam is: most reliable
        avoid.append(corner_box(facecam, sw, sh, facecam_size))
    elif facecam != "none" and face_detect:
        avoid.extend(find_faces(list(gray.astype(np.uint8)), sw))

    def up(b: Box) -> Box:
        return (_even(b[0] * sx), _even(b[1] * sy), _even(b[2] * sx), _even(b[3] * sy))

    return Layout(frame_w, frame_h, up(rect_small), [up(b) for b in avoid], origin)


# ------------------------------------------------------------------ step 2: crop for one clip

def energy_map(frames: list[np.ndarray]) -> np.ndarray:
    """Where is the action? temporal change + edge detail, on the sampled frames."""
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)
    temporal = gray.std(axis=0) if len(gray) > 1 else np.zeros_like(gray[0])
    edges = np.mean([np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)) for g in gray], axis=0)
    e = temporal / (temporal.max() + 1e-6) * 0.7 + edges / (edges.max() + 1e-6) * 0.3
    return cv2.GaussianBlur(e, (0, 0), 6)


def choose_crop(layout: Layout, energy: np.ndarray | None, aspect: "float | tuple[float, float]" = 9 / 16,
                energy_scale: tuple[float, float] | None = None) -> Box:
    """Pick the crop window (source pixels). aspect = width / height of the window."""
    rects = _exclude([layout.rect], layout.avoid[0]) if layout.avoid else [layout.rect]
    for box in layout.avoid[1:]:
        rects = _exclude(rects, box)
    if not rects:  # facecam in the middle of everything: fall back to the plain rect
        rects = [layout.rect]

    if energy is not None:
        eh, ew = energy.shape
        sx = layout.frame_w / ew
        sy = layout.frame_h / eh
        sat = cv2.integral(energy.astype(np.float64))
        total = float(sat[-1, -1]) + 1e-9

        def e_of(x: int, y: int, w: int, h: int) -> float:
            x0, y0 = max(0, int(x / sx)), max(0, int(y / sy))
            x1, y1 = min(ew, int((x + w) / sx)), min(eh, int((y + h) / sy))
            if x1 <= x0 or y1 <= y0:
                return 0.0
            return float(sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]) / total
    else:
        def e_of(x: int, y: int, w: int, h: int) -> float:
            return 0.0

    best, best_score = None, -1.0
    area_w = 3.0 if isinstance(aspect, tuple) else 0.6      # hybrid: show as much game picture as possible
    rect_area = layout.rect[2] * layout.rect[3]
    for (rx, ry, rw, rh) in rects:
        asp = min(max(rw / rh, aspect[0]), aspect[1]) if isinstance(aspect, tuple) else aspect
        if rh * asp <= rw:
            ch, cw = rh, rh * asp
        else:
            cw, ch = rw, rw / asp
        cw, ch = _even(cw), _even(ch)
        slack_x, slack_y = rw - cw, rh - ch
        steps = 24
        for i in range(steps + 1):
            fx = i / steps
            x = rx + int(slack_x * fx) if slack_x > 0 else rx
            y = ry + int(slack_y * fx) if slack_y > 0 and slack_x <= 0 else ry
            center_dev = abs(fx - 0.5) * 2 if (slack_x > 0 or slack_y > 0) else 0.0
            score = e_of(x, y, cw, ch) + area_w * (cw * ch / max(rect_area, 1)) + 0.12 * (1 - center_dev)
            if score > best_score:
                best_score, best = score, (_even(x), _even(y), cw, ch)
    assert best is not None
    return best


def draw_debug(frame: np.ndarray, layout: Layout, crop: Box | None) -> np.ndarray:
    h, w = frame.shape[:2]
    sx, sy = w / layout.frame_w, h / layout.frame_h

    def r(b: Box, color, thick=3):
        cv2.rectangle(frame, (int(b[0] * sx), int(b[1] * sy)),
                      (int((b[0] + b[2]) * sx), int((b[1] + b[3]) * sy)), color, thick)

    r(layout.rect, (0, 255, 0))
    for b in layout.avoid:
        r(b, (0, 0, 255))
    if crop:
        r(crop, (0, 255, 255), 4)
    return frame


def find_cam(L: Layout, video, duration: float, facecam: str = "auto",
             facecam_size: tuple[float, float] = (0.27, 0.38)) -> Box | None:
    """The webcam box for the streamer layout: the one the AI marked, a corner the owner configured, or a face
    that stays in a corner of the picture. None when the stream has no webcam."""
    if L.cam:
        return L.cam
    facecam = (facecam or "auto").lower()
    if facecam == "none":
        return None
    if facecam in CORNERS:
        return corner_box(facecam, L.frame_w, L.frame_h, facecam_size)
    try:
        times = list(np.linspace(duration * 0.1, duration * 0.9, 12)) if duration > 10 else [duration / 2]
        frames = sample_frames(video, times, 640)
        if len(frames) < 4:
            return None
        sh, sw = frames[0].shape[:2]
        sx, sy = L.frame_w / sw, L.frame_h / sh
        found = find_faces([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames], sw)
        if found:
            b = found[0]
            return (_even(b[0] * sx), _even(b[1] * sy), _even(b[2] * sx), _even(b[3] * sy))
    except Exception:
        pass
    return None
