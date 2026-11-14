from pathlib import Path

import pytest

from featureclip import captions, judge, narrate, uploader
from featureclip.sources import Source


# ------------------------------------------------------------------ judge rules

def raw(**kw):
    base = {"feature_shown": True, "category": "vehicle", "feature_name": "Realistic car damage",
            "what_happens": "A car crumples on impact.", "confidence": 85, "interest": 80,
            "hook_title": "Car damage is wild", "start_offset": 5, "end_offset": 30, "reason": "x"}
    base.update(kw)
    return base


def test_judge_accepts_a_good_feature(cfg):
    j = judge.normalize(raw(), 30, 15, 40)
    assert judge.decide(j, cfg) == (True, "ok")
    assert (j["start_offset"], j["end_offset"]) == (5.0, 30.0)


@pytest.mark.parametrize("kw,fragment", [
    ({"category": "cutscene"}, "rejected category"),
    ({"category": "story_spoiler"}, "rejected category"),
    ({"category": "glitch_exploit"}, "rejected category"),
    ({"category": "mod_or_cheat"}, "rejected category"),
    ({"category": "made_up_category"}, "rejected category"),   # unknown -> nothing_notable
    ({"feature_shown": False}, "no feature"),
    ({"confidence": 30}, "confidence"),
    ({"interest": 10}, "interest"),
    ({"feature_name": ""}, "missing"),
])
def test_judge_hard_rules(cfg, kw, fragment):
    ok, why = judge.decide(judge.normalize(raw(**kw), 30, 15, 40), cfg)
    assert not ok and fragment in why


def test_bad_offsets_fall_back_to_whole_window():
    j = judge.normalize(raw(start_offset=28, end_offset=29), 30, 15, 40)
    assert (j["start_offset"], j["end_offset"]) == (0.0, 30.0)
    j = judge.normalize(raw(start_offset="abc", end_offset=None), 30, 15, 40)
    assert j["end_offset"] - j["start_offset"] >= 15


def test_dedupe_keeps_best_per_feature():
    a = judge.normalize(raw(interest=70), 30, 15, 40)
    b = judge.normalize(raw(interest=90), 30, 15, 40)
    c = judge.normalize(raw(feature_name="Rain on windshield"), 30, 15, 40)
    assert set(judge.dedupe([(1, a), (2, b), (3, c)], 8)) == {2, 3}
    assert len(judge.dedupe([(1, a), (2, b), (3, c)], 1)) == 1


# ------------------------------------------------------------------ narration script

FACTS = "Realistic car damage. A car crumples on impact. wait look at this"


def test_script_validation_rules():
    ok = "Look at what happens when this car hits the wall. " * 3
    assert narrate.validate_script(ok, FACTS, 10, 60) == []
    assert any("too short" in p for p in narrate.validate_script("Short one.", FACTS, 10, 60))
    assert any("too long" in p for p in narrate.validate_script("word " * 100, FACTS, 10, 60))
    assert any("call-to-action" in p for p in narrate.validate_script(ok + " Like and subscribe!", FACTS, 10, 80))
    assert any("number '42'" in p for p in narrate.validate_script(ok + " It takes 42 hits.", FACTS, 10, 80))
    assert narrate.validate_script(ok + " It takes 42 hits.", FACTS + " 42", 10, 80) == []


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def json(self, prompt, **kw):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def test_write_script_retries_with_feedback(cfg):
    j = judge.normalize(raw(), 30, 15, 40)
    good = ("Wait, look at what this car just did. The front folds inward on impact. "
            "The glass cracks across the windshield. Every crash looks different. "
            "It is a small detail that makes the world feel real.")
    llm = FakeLLM([{"script": "Too short.", "on_screen_hook": "x"},
                   {"script": good, "on_screen_hook": "CAR DAMAGE"}])
    out = narrate.write_script(llm, cfg, j, "wait look at this", 24)
    assert out["script"] == good
    assert "too short" in llm.prompts[1]          # the problems were fed back to the model


def test_write_script_gives_up(cfg):
    j = judge.normalize(raw(), 30, 15, 40)
    llm = FakeLLM([{"script": "No.", "on_screen_hook": ""}] * 3)
    with pytest.raises(ValueError):
        narrate.write_script(llm, cfg, j, "", 24)


def test_synthesize_dummy_speeds_up_when_too_long(cfg, tmp_path):
    text = "This is a fairly long sentence to speak out loud for the test run. " * 3
    n1 = narrate.synthesize(cfg, text, tmp_path / "a.wav")
    n2 = narrate.synthesize(cfg, text, tmp_path / "b.wav", max_duration=n1.duration / 1.1)
    assert 0.85 * n1.duration / 1.1 < n2.duration < 1.02 * n1.duration / 1.1
    assert len(n1.words) == len(text.split())
    assert all(a < b for a, b, _ in n1.words) and n1.words[-1][1] <= n1.duration + 0.01


# ------------------------------------------------------------------ captions

def test_ass_structure(cfg):
    words = [(i * 0.4, i * 0.4 + 0.35, w) for i, w in enumerate("one two three four five six seven".split())]
    ass = captions.build_ass(words, "Car damage is wild", cfg, offset=0.4)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert ass.count("Dialogue:") == len(words) + 1                # one per word + the hook
    assert "CAR DAMAGE IS WILD" in ass
    assert "{\\c&H00E6FF&}one{\\c&HFFFFFF&} two three" in ass       # active word highlighted
    assert "{" not in captions._esc("a{b}c")


# ------------------------------------------------------------------ upload metadata

def test_title_and_body(cfg):
    t = uploader.build_title("x" * 200, "feat", " #Shorts")
    assert len(t) <= 100 and t.endswith(" #Shorts")
    body = uploader.build_body("T", "D", cfg)
    assert body["status"]["privacyStatus"] == "private"            # safe default
    assert body["status"]["containsSyntheticMedia"] is True
    assert body["status"]["selfDeclaredMadeForKids"] is False
    assert body["snippet"]["categoryId"] == "20"


def test_description_credits_streamer(cfg):
    j = judge.normalize(raw(), 30, 15, 40)
    d = uploader.build_description(j, "ExampleStreamer", "https://x.test/@e", cfg)
    assert "ExampleStreamer" in d and "https://x.test/@e" in d and "AI-generated" in d and "#Shorts" in d


# ------------------------------------------------------------------ permission gate

def test_permission_gate():
    with pytest.raises(PermissionError):
        Source(id="a", streamer="s", permission="pending").check_permission()
    with pytest.raises(PermissionError):
        Source(id="a", streamer="s", permission="granted", permission_proof="  ").check_permission()
    Source(id="a", streamer="s", permission="granted", permission_proof="DM 2026-11-16").check_permission()
