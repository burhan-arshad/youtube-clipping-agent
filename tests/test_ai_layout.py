"""AI layout: the vision model is faked; geometry, validation and the safety net are real."""
from featureclip import ai_layout, layout as lay


class FakeLLM:
    def __init__(self, answer):
        self.answer, self.calls, self.images, self.counts = answer, 0, 0, []

    def available(self):
        return True

    def json(self, prompt, system="", images=None, temperature=0.4):
        self.calls += 1
        self.images = len(images or [])
        self.counts.append(self.images)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


SIDEBAR = {"gameplay": [0, 260, 1000, 830],
           "overlays": [{"kind": "logo", "box": [930, 260, 1000, 420]},
                        {"kind": "chat", "box": [0, 840, 500, 1000]},        # outside the gameplay box -> ignored
                        {"kind": "facecam", "box": [550, 0, 1000, 250]}],     # outside -> ignored
           "layout_type": "game_with_sidebar", "notes": "ok"}


def test_parse_sidebar_layout():
    rect, avoid, info = ai_layout.parse(SIDEBAR, 1920, 1080)
    assert 500 <= rect[0] <= 515 and rect[1] >= 0 and abs(rect[2] - 0.564 * 1920) < 40 and rect[3] >= 1060 - 4
    assert len(avoid) == 1 and info["kinds"] == ["logo"]                  # only the logo touches the game picture


def test_parse_rejects_nonsense():
    assert ai_layout.parse({}, 640, 360) is None
    assert ai_layout.parse({"gameplay": [0, 0, 100, 100]}, 640, 360) is None          # < 20% of the frame
    assert ai_layout.parse({"gameplay": "a"}, 640, 360) is None
    big = {"gameplay": [0, 0, 1000, 1000], "overlays": [{"kind": "x", "box": [0, 0, 1000, 800]}]}
    assert ai_layout.parse(big, 640, 360)[1] == []                         # half-screen "overlay" = mistake


def test_detect_sends_three_screenshots_and_falls_back(test_video, tmp_path):
    llm = FakeLLM({"gameplay": [0, 0, 1000, 1000], "overlays": [{"kind": "facecam", "box": [0, 750, 300, 1000]}]})
    L = ai_layout.detect(llm, test_video, 70, 640, 360, tmp_path / "t")
    assert llm.counts == [3, 1] and L.origin == "ai" and len(L.avoid) == 1
    x, y, w, h = L.avoid[0]
    assert x < 500 and y == 0 and x + w >= 636                              # top-right corner, a little padding
    assert ai_layout.detect(FakeLLM(RuntimeError("boom")), test_video, 70, 640, 360, tmp_path / "t2") is None


def test_detect_layout_uses_ai_and_crop_avoids_overlay(test_video, tmp_path):
    llm = FakeLLM({"gameplay": [0, 0, 1000, 1000], "overlays": [{"kind": "chat", "box": [0, 800, 1000, 1000]}]})
    L = lay.detect_layout(test_video, 70, 640, 360, llm=llm, tmp=tmp_path / "t")
    assert L.origin == "ai"
    crop = lay.choose_crop(L, None, (1.0, 1.7))
    assert crop[0] + crop[2] <= L.avoid[0][0] + 2                            # crop ends where the chat starts


def test_manual_settings_skip_the_ai(test_video, tmp_path):
    llm = FakeLLM({})
    L = lay.detect_layout(test_video, 70, 640, 360, llm=llm, tmp=tmp_path, crop_frac=[0, 0, 0.5, 1])
    assert llm.calls == 0 and L.origin == "manual"
    L = lay.detect_layout(test_video, 70, 640, 360, llm=llm, tmp=tmp_path, facecam="top-right")
    assert llm.calls == 0
