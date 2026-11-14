import numpy as np

from featureclip import candidates, layout as lay, signals
from featureclip.ffutil import probe


def test_signals_find_the_two_events(test_video):
    info = probe(test_video)
    sig = signals.compute_signals(str(test_video), info["duration"], 2)
    comb = candidates.combine(sig, 1.0, 0.8, 0.6)
    cands = candidates.pick_candidates(comb, info["duration"], top_k=2, skip_start=10, min_gap=20)
    peaks = sorted(c.peak for c in cands)
    assert len(peaks) == 2
    assert abs(peaks[0] - 25) <= 3 and abs(peaks[1] - 50) <= 3
    for c in cands:
        assert 0 <= c.start < c.end <= info["duration"] + 0.01
        assert c.end - c.start <= 30.01


def test_layout_cuts_static_overlays(test_video):
    info = probe(test_video)
    L = lay.detect_layout(str(test_video), info["duration"], info["width"], info["height"],
                          n_frames=16, face_detect=True)
    x, y, w, h = L.rect
    k = info["width"] / 1920
    assert L.origin == "auto"
    assert x >= 8 * k and y >= 8 * k                       # static border removed
    assert y + h <= info["height"] - 60 * k                # static HUD strip removed
    assert w * h > 0.5 * info["width"] * info["height"]    # but most of the picture is kept


def test_manual_crop_wins(test_video):
    info = probe(test_video)
    L = lay.detect_layout(str(test_video), info["duration"], info["width"], info["height"],
                          manual=[11, 21, 301, 201])
    assert L.origin == "manual" and L.rect == (10, 20, 300, 200)


def test_crop_is_vertical_and_avoids_facecam():
    L = lay.Layout(1920, 1080, (0, 0, 1920, 1080), [(900, 0, 400, 500)])
    x, y, w, h = lay.choose_crop(L, None, 9 / 16)
    assert abs(w / h - 9 / 16) < 0.01
    assert not lay._inter((x, y, w, h), (900, 0, 400, 500))


def test_crop_follows_the_action():
    L = lay.Layout(1920, 1080, (0, 0, 1920, 1080), [])
    energy = np.zeros((108, 192), dtype=np.float32)
    energy[:, 150:180] = 1.0                      # action on the right side
    x, _, w, _ = lay.choose_crop(L, energy, 9 / 16)
    assert x + w / 2 > 1100


def test_facecam_clustering():
    # the same face in the same place in most frames -> a facecam; a wandering face -> ignored
    fixed = [[(500, 20, 50, 50)] for _ in range(10)]
    wandering = [[(40 * i, 100, 50, 50)] for i in range(10)]
    assert len(lay.cluster_faces(fixed, 10)) == 1
    assert lay.cluster_faces(wandering, 10) == []
    box = lay.facecam_box((500, 20, 50, 50), 640, 360)
    assert box[1] == 0 and box[0] < 500 < box[0] + box[2]    # snapped to the top edge


def test_crop_never_uses_facecam_pixels_even_when_cam_is_central():
    L = lay.Layout(1920, 1080, (0, 0, 1920, 1080), [(0, 0, 1920, 300)])   # cam banner across the top
    x, y, w, h = lay.choose_crop(L, None, 9 / 16)
    assert y >= 300
