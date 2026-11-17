import base64
import io

import numpy as np
import pytest

from featureclip import judge, metadata, narrate, tts
from featureclip.config import load_config
from featureclip.transcribe import Transcript


@pytest.fixture
def cfg(tmp_path):
    c = load_config(None, root=tmp_path)
    c.data["narration"]["engine"] = "dummy"
    return c


def test_title_suffix_is_not_a_hashtag_problem_but_other_hashtags_are():
    assert metadata.validate({"title": "GTA 6 cars crumple like real metal #Shorts", "description": "x" * 100}, "GTA 6") == []
    p = metadata.validate({"title": "GTA 6 #cars crumple #Shorts", "description": "x" * 100}, "GTA 6")
    assert any("hashtag" in x for x in p)


def test_thumbnail_text_is_short_and_has_no_hashtag():
    assert metadata.thumb_line({"thumb_text": "REAL CRASH DAMAGE"}, "", "t") == "REAL CRASH DAMAGE"
    t = metadata.thumb_line(None, "", "GTA 6: Realistic vehicle damage physics you probably missed #Shorts")
    assert "#" not in t and len(t) <= 34 and not t.endswith(" ")


def test_description_has_names_only_no_links(cfg):
    cfg.data["branding"]["name"] = "Leonida Files"
    lines = metadata.credit_lines(cfg, "Ducky Bhai")
    assert lines[0] == "Footage from Ducky Bhai's stream." and lines[-1] == "Channel: Leonida Files"
    assert not any("http" in l for l in lines)


def test_beats_validation_catches_overlap_and_late_end(cfg):
    ok = [{"at": 0.5, "text": "GTA 6 just did this."}, {"at": 4.0, "text": "Watch the bumper crumple."},
          {"at": 9.0, "text": "Did you notice it?"}]
    assert narrate.validate_beats(ok, 15, cfg) == []
    bad = [{"at": 0.5, "text": "One two three four five six seven eight nine ten eleven twelve."},
           {"at": 2.0, "text": "Too early."}]
    assert any("starts at" in p for p in narrate.validate_beats(bad, 15, cfg))
    assert any("clip is only" in p for p in narrate.validate_beats(ok, 6, cfg))


def test_align_words_maps_script_to_what_whisper_heard():
    script = "Watch this car, it crumples!".split()
    heard = [(0.1, 0.4, "Watch"), (0.4, 0.7, "this"), (0.7, 1.0, "car"), (1.2, 1.9, "crumples")]
    pos = narrate.align_words(script, heard)
    assert pos is not None and pos[0][0] == 0.1 and pos[4] == (1.2, 1.9)
    assert pos[2][1] <= pos[3][0] + 1e-6 and pos[3][1] <= pos[4][0] + 1e-6                  # "it" was not heard: interpolated between neighbours
    assert narrate.align_words(["a", "b", "c", "d"], [(0, 1, "zzz")]) is None


def test_local_voice_places_each_beat_on_its_moment(cfg, tmp_path):
    beats = [{"at": 0.5, "text": "GTA 6 just did this."}, {"at": 8.0, "text": "Here comes the crash."}]
    n = narrate.synthesize(cfg, "x", tmp_path / "n.wav", beats=beats, lead=0.4)
    starts = {w[2]: w[0] for w in n.words}
    assert starts["Here"] >= 8.0 - 0.4 - 0.05            # second line waits for its moment (minus the lead-in)
    assert starts["GTA"] < 1.0


def test_cloud_voice_cut_into_beats_and_aligned(cfg, monkeypatch, tmp_path):
    cfg.data["narration"]["engine"] = "gemini"
    sr = 24000
    audio = (0.1 * np.sin(np.arange(sr * 6) / 20)).astype(np.float32)           # 6 s of tone
    monkeypatch.setattr(tts, "gemini_speak", lambda c, text, **k: (audio, sr))
    words = "Watch this car now <short pause> it crumples hard".split()
    heard = [(0.2 + i * 0.6, 0.2 + i * 0.6 + 0.5, w) for i, w in enumerate(tts.strip_tags(" ".join(words)).split())]
    import featureclip.transcribe as T
    monkeypatch.setattr(T, "transcribe_file", lambda *a, **k: Transcript(words=heard, available=True))
    beats = [{"at": 0.5, "text": "Watch this car now"}, {"at": 9.0, "text": "<short pause> it crumples hard"}]
    n = narrate.synthesize(cfg, "x", tmp_path / "n.wav", beats=beats, lead=0.4)
    assert n.aligned and [w[2] for w in n.words] == "Watch this car now it crumples hard".split()
    second = next(w for w in n.words if w[2] == "it")
    assert second[0] >= 9.0 - 0.4 - 0.1                    # moved to its moment, not left at its natural time (~2.6 s)
    assert n.duration > 9.0 - 0.4


def test_cloud_voice_falls_back_to_offline_when_it_fails(cfg, monkeypatch, tmp_path):
    cfg.data["narration"]["engine"] = "gemini"
    monkeypatch.setattr(tts, "gemini_speak", lambda *a, **k: (_ for _ in ()).throw(tts.TTSError("quota")))
    n = narrate.synthesize(cfg, "A short test line.", tmp_path / "n.wav")
    assert n.duration > 0 and (tmp_path / "n.wav").is_file()


def test_gemini_speak_request_shape_and_wav_decoding(cfg, tmp_path, monkeypatch):
    import soundfile as sf
    cfg.data["narration"]["engine"] = "gemini"
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    cfg.root = tmp_path
    buf = io.BytesIO()
    sf.write(buf, np.zeros(24000, dtype=np.float32) + 0.1, 24000, format="WAV")
    seen = {}

    class R:
        status_code = 200
        text = ""
        def json(self):
            return {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "audio/wav",
                    "data": base64.b64encode(buf.getvalue()).decode()}}]}}]}

    def post(url, json=None, headers=None, timeout=0):
        seen.update(url=url, body=json, headers=headers)
        return R()
    audio, sr = tts.gemini_speak(cfg, "Hello there", voice="Fenrir", style="excited", post=post)
    assert sr == 24000 and len(audio) == 24000
    assert seen["url"].endswith("gemini-3.8-flash-tts:generateContent")
    part = seen["body"]["contents"][0]["parts"][0]
    assert part["text"] == "Hello there" and part["speech_metadata"]["style"] == "excited"
    assert seen["body"]["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert seen["body"]["generationConfig"]["speechConfig"]["voiceConfig"]["voice"] == "Fenrir"


def test_keyframes_survive_one_unreadable_frame_and_fail_when_all_bad(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_grab(video, t, path, width, crop=None):
        calls["n"] += 1
        if t > 28:
            raise RuntimeError("past the end")
        path.write_bytes(b"jpg")
    monkeypatch.setattr(judge, "grab_frame", fake_grab)
    imgs, times = judge.grab_keyframes("v.mp4", 0, 30, 4, 512, tmp_path / "k")
    assert len(imgs) == 4 and max(times) <= 28            # last frame was retried earlier in the clip
    monkeypatch.setattr(judge, "grab_frame", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad")))
    with pytest.raises(RuntimeError):
        judge.grab_keyframes("v.mp4", 0, 30, 4, 512, tmp_path / "k2")


def test_judge_key_moment_shapes_the_clip():
    raw = {"feature_shown": True, "category": "vehicle", "feature_name": "Car damage", "what_happens": "x",
           "visible_evidence": ["a", "b"], "confidence": 80, "interest": 80, "key_moment_offset": 18,
           "start_offset": 16, "end_offset": 20, "timeline": [{"frame": 1, "what": "driving"}]}
    j = judge.normalize(raw, 30, 15, 40)
    assert j["start_offset"] <= 15.5 and j["end_offset"] >= 22.5          # sees it coming, sees the result
    assert j["end_offset"] - j["start_offset"] >= 15 and j["timeline"] == ["driving"]


def test_voice_endpoints(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import featureclip.web.app as A
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    c = TestClient(A.create_app(tmp_path))
    v = c.get("/api/voices").json()
    assert any(x["id"] == "Fenrir" for x in v["prebuilt"]) and v["library"] == []
    r = c.post("/api/voice-test", json={"voice": "Fenrir"})
    assert r.status_code == 400 and "Gemini key" in r.json()["detail"]
    fields = {f["key"] for f in c.get("/api/settings").json()}
    assert {"narration.gemini_voice", "narration.gemini_style", "game.hints", "branding.name",
            "env:ELEVENLABS_API_KEY"} <= fields
