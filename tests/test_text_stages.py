from pathlib import Path

import pytest

from featureclip import captions, judge, narrate, uploader
from featureclip.sources import Source


# ------------------------------------------------------------------ judge rules

def raw(**kw):
    base = {"feature_shown": True, "category": "vehicle", "feature_name": "Realistic car damage",
            "what_happens": "A car crumples on impact.", "confidence": 85, "interest": 80,
            "visible_evidence": ["the bonnet folds in", "glass cracks"], "real_person_or_overlay_visible": False,
            "hook_title": "Car damage is wild", "start_offset": 5, "end_offset": 30, "reason": "x"}
    base.update(kw)
    return base


def test_judge_accepts_a_good_feature(cfg):
    j = judge.normalize(raw(), 30, 15, 40)
    assert judge.decide(j, cfg) == (True, "ok")
    assert (j["start_offset"], j["end_offset"]) == (5.0, 30.0)


@pytest.mark.parametrize("kw,fragment", [
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
    assert any("game name" in p for p in narrate.validate_script(ok, FACTS, 10, 60, game="GTA 6"))
    assert narrate.validate_script("GTA VI " + ok, FACTS, 10, 60, game="GTA 6") == []
    assert any("question" in p for p in narrate.validate_script(ok, FACTS, 10, 60, need_question=True))
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
    good = ("Wait, look at what this car just did in GTA 6. The front folds inward on impact. "
            "The glass cracks across the windshield. Every crash looks different. "
            "The metal bends in a new way each time. "
            "It is a small detail that makes the whole world feel real. "
            "Which detail did you spot first?")
    llm = FakeLLM([{"script": "Too short.", "on_screen_hook": "x"},
                   {"script": good, "on_screen_hook": "CAR DAMAGE"}])
    out = narrate.write_script(llm, cfg, j, "wait look at this", 24)
    assert out["script"] == good
    assert "too short" in llm.prompts[1]          # the problems were fed back to the model


def test_write_script_gives_up(cfg):
    j = judge.normalize(raw(), 30, 15, 40)
    llm = FakeLLM([{"script": "No.", "on_screen_hook": ""}] * 4)
    with pytest.raises(ValueError):
        narrate.write_script(llm, cfg, j, "", 24)


def test_fit_length_trims_middle_sentences():
    sents = ["Wait, did you see that."] + [f"Detail number here with some more words in it {i}." for i in range(8)] + ["Which part did you spot first?"]
    text = " ".join(sents)
    out = narrate._fit_length(text, 30, 60)
    assert out and 27 <= narrate.word_count(out) <= 60
    assert out.startswith("Wait") and out.endswith("first?")


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
    assert "CAR DAMAGE IS" in ass and "WILD" in ass
    assert "{\\c&H00E6FF&\\fscx100\\fscy100\\t(0,90,\\fscx112\\fscy112)}one{\\c&HFFFFFF&\\fscx100\\fscy100} two three" in ass  # active word highlighted + pop
    assert "{" not in captions._esc("a{b}c")


# ------------------------------------------------------------------ upload metadata

def test_hooks_vary_and_title_goes_in_blur_strip(cfg):
    import random
    rng = random.Random(1)
    picks = {narrate._pick("hook", narrate.HOOK_STYLES, rng) for _ in range(6)}
    assert len(picks) >= 4                                    # not the same hook every time
    ass = captions.build_ass([(0, 0.3, "hi")], "GTA 6: Cops can chase you", cfg, title_box=(300, 260, 420), duration=20, handle="@DetailRadar")
    assert "Title,,0,0,0,,{\\an5\\pos(540,300)" in ass and "0:00:20.00" in ass
    assert "Handle,,0,0,0,,{\\an5\\pos(540,420)}@DetailRadar" in ass
    from featureclip import compose
    assert compose.title_area((0, 0, 1700, 1000), "hybrid", 1080, 1920) is not None
    assert compose.title_area((0, 0, 1700, 1000), "fill", 1080, 1920) is None


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
    assert "ExampleStreamer" in d and "http" not in d and "AI-generated" in d and "#Shorts" in d


# ------------------------------------------------------------------ source resolving

def test_resolve_source_link_and_unknown(tmp_path):
    from featureclip.sources import resolve_source
    s = resolve_source("https://youtu.be/x", {})
    assert s.url == "https://youtu.be/x" and s.id.startswith("web-")
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    assert resolve_source(str(f), {}).path == str(f)
    with pytest.raises(KeyError):
        resolve_source("nope", {})

# ------------------------------------------------------------------ hook fits, music, script length

def test_hook_fits_inside_the_frame():
    from featureclip.captions import fit_hook
    txt, size = fit_hook("HOSPITAL RESPAWN RECOVERY MECHANIC EXPLAINED PROPERLY", 1080, 96)
    lines = txt.split("\\N")
    assert 1 <= len(lines) <= 3
    for ln in lines:
        assert len(ln) * 0.64 * size <= 1080 - 160 + 1, (ln, size)
    short, s2 = fit_hook("WILD CAR DAMAGE", 1080, 96)
    assert s2 >= 70


def test_script_must_fill_the_clip():
    from featureclip import narrate
    class C:
        def get(self, k, d=None):
            return {"narration.lead_in": 0.4, "narration.words_per_second": 2.5}.get(k, d)
    lo, hi = narrate.target_words(30, C())
    assert hi == 71 and lo == int(hi * 0.85)


def test_generated_music_is_valid_and_mixed(cfg, test_video, tmp_path):
    import numpy as np
    from featureclip import compose, music
    from featureclip.ffutil import probe, run
    x = music.render_track(0, 12.0, seed=3)
    assert x.dtype == np.float32 and 0.05 < float(np.abs(x).max()) <= 1.0 and not np.isnan(x).any()
    cfg.data["audio"]["music"] = True
    track = music.pick_track(cfg, tmp_path, "1-1", tmp_path)
    assert track and track.is_file()
    wav = tmp_path / "n.wav"
    run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=6", str(wav)])
    out = tmp_path / "o.mp4"
    compose.render(test_video, 5, 10, (0, 0, 640, 360), wav, None, out, cfg, music_path=track)
    info = probe(out)
    assert info["has_audio"] and 9 <= info["duration"] <= 11


def test_cutscene_is_allowed_by_default_but_can_be_blocked(cfg):
    j = judge.normalize(raw(category="cutscene"), 30, 15, 40)
    assert judge.decide(j, cfg)[0]
    cfg.data["judge"]["allow_cutscenes"] = False
    cfg.data["judge"]["reject_categories"].append("cutscene")
    assert not judge.decide(j, cfg)[0]


def test_streamer_visible_and_missing_evidence_are_rejected(cfg):
    j = judge.normalize(raw(real_person_or_overlay_visible=True), 30, 15, 40)
    ok, why = judge.decide(j, cfg)
    assert not ok and "streamer" in why
    j = judge.normalize(raw(visible_evidence=["only one thing"]), 30, 15, 40)
    assert judge.decide(j, cfg) == (False, "no concrete visible evidence")


def test_strictness_presets(cfg):
    j = judge.normalize(raw(confidence=50, interest=45), 30, 15, 40)
    cfg.data["judge"]["strictness"] = "loose"
    assert judge.decide(j, cfg)[0]
    cfg.data["judge"]["strictness"] = "strict"
    assert not judge.decide(j, cfg)[0]
    cfg.data["judge"]["min_confidence"] = 10       # explicit numbers win over the preset
    cfg.data["judge"]["min_interest"] = 10
    assert judge.decide(j, cfg)[0]


def test_llm_limits_lookup():
    from featureclip.llm_limits import limits_for
    assert limits_for("gemini-3.5-flash-lite") == {"rpm": 15, "tpm": 250000, "rpd": 500}
    assert limits_for("gemini-3.8-flash-preview-05")["rpd"] == 20
    assert limits_for("unknown-model") is None
