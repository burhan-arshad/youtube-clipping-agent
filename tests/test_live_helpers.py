from pathlib import Path

from featureclip import judge
from featureclip.ffutil import grab_frame


def test_feature_key_ignores_order_and_case():
    assert judge.feature_key("Car Damage realistic") == judge.feature_key("realistic car damage")


def test_grab_frame_steps_back_at_end_of_file(test_video, tmp_path):
    out = grab_frame(test_video, 69.9999, tmp_path / "f.jpg", 320)   # 70 s file, asks for the very end
    assert Path(out).stat().st_size > 0
