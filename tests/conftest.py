import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from featureclip.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def test_video(tmp_path_factory) -> Path:
    """Small synthetic 'stream': loud + visual events at 25 s and 50 s."""
    from make_test_video import make
    p = tmp_path_factory.mktemp("vid") / "stream.mp4"
    make(str(p), duration=70, w=640, h=360, fps=15, events=(25, 50))
    return p


@pytest.fixture
def cfg(tmp_path):
    c = load_config(root=tmp_path)
    c.data["detect"]["skip_start"] = 10
    c.data["narration"]["engine"] = "dummy"
    c.data["whisper"]["model"] = "tiny.en"
    return c
