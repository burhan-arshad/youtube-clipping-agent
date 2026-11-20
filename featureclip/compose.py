"""One ffmpeg call: cut -> crop to gameplay -> 9:16 -> captions + voice-over -> loudness -> mp4."""
from __future__ import annotations

from pathlib import Path

from .ffutil import run

Box = tuple[int, int, int, int]


def title_area(crop: Box, mode: str, width: int, height: int) -> tuple[int, int, int] | None:
    """(heading centre_y, heading max height, handle centre_y) of the blurred strip above the gameplay block
    in hybrid mode, else None. The channel handle sits right above the gameplay, under the heading."""
    if mode != "hybrid":
        return None
    _, _, w, h = crop
    block_h = min(int(width * h / w), 1500) // 2 * 2
    top = int((height - block_h) * 0.35)
    safe = 150                                    # keep clear of the YouTube Shorts top bar
    if top - safe < 120:
        return None
    handle_h = 64
    area = top - safe - 20 - handle_h
    return safe + area // 2, area, top - 20 - handle_h // 2


def video_filter(crop: Box, mode: str, width: int, height: int, fps: int, ass_name: str | None,
                 block_aspect: float = 0.8) -> str:
    x, y, w, h = crop
    base = f"[0:v]crop={w}:{h}:{x}:{y},setsar=1"
    if mode == "hybrid":
        block_h = min(int(width * h / w), 1500) // 2 * 2
        top = int((height - block_h) * 0.35)
        graph = (
            f"{base},split[a][b];"
            f"[b]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},"
            f"boxblur=26:3,eq=brightness=-0.18[bg];"
            f"[a]scale={width}:{block_h}:flags=lanczos[fg];"
            f"[bg][fg]overlay=0:{top},fps={fps}"
        )
    else:  # fill
        graph = f"{base},scale={width}:{height}:flags=lanczos,fps={fps}"
    if ass_name:
        graph += f",ass={ass_name}"
    return graph + "[v]"


def audio_filter(cfg, dur: float, has_game_audio: bool, lead_in: float, music: bool = False) -> str:
    lufs = float(cfg.get("audio.target_lufs", -14))
    lead_ms = int(lead_in * 1000)
    narr = f"[1:a]adelay={lead_ms}|{lead_ms},aresample=48000,apad"
    if cfg.get("audio.game_audio", "off") == "duck" and has_game_audio:
        duck = float(cfg.get("audio.duck_db", -22))
        graph = (
            f"{narr},asplit[n1][n2];"
            f"[0:a]aresample=48000,volume={duck}dB[g];"
            f"[g][n1]sidechaincompress=threshold=0.04:ratio=10:attack=20:release=350[gd];"
            f"[gd][n2]amix=inputs=2:duration=longest:normalize=0,"
        )
    elif music:
        mdb = float(cfg.get("audio.music_db", -22))
        fo = max(0.0, dur - 1.8)
        graph = (
            f"{narr},asplit[n1][n2];"
            f"[2:a]aresample=48000,volume={mdb}dB,afade=t=in:st=0:d=1.2,afade=t=out:st={fo:.3f}:d=1.8[m0];"
            f"[m0][n1]sidechaincompress=threshold=0.015:ratio=12:attack=10:release=650[md];"
            f"[n2][md]amix=inputs=2:duration=longest:normalize=0,"
        )
    else:
        graph = f"{narr},"
    graph += f"atrim=0:{dur:.3f},loudnorm=I={lufs}:TP=-1.5:LRA=11[a]"
    return graph


def render(video: str | Path, start: float, dur: float, crop: Box, narration_wav: Path,
           ass_path: Path | None, out_path: Path, cfg, has_game_audio: bool = True,
           music_path: Path | None = None) -> Path:
    """Render one Short. ffmpeg runs inside the clip folder so the ASS filename needs no escaping."""
    out_path = Path(out_path)
    work = ass_path.parent if ass_path else out_path.parent
    width, height = int(cfg.get("video.width")), int(cfg.get("video.height"))
    fps = int(cfg.get("video.fps"))
    vf = video_filter(crop, cfg.get("layout.mode", "fill"), width, height, fps,
                      ass_path.name if ass_path else None,
                      block_aspect=float(cfg.get("layout.hybrid_aspect", 1.33)))
    use_music = bool(music_path) and cfg.get("audio.game_audio", "off") != "duck"
    af = audio_filter(cfg, dur, has_game_audio, float(cfg.get("narration.lead_in", 0.4)), music=use_music)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(Path(video).resolve()),
        "-i", str(Path(narration_wav).resolve()),
    ]
    if use_music:
        cmd += ["-stream_loop", "-1", "-i", str(Path(music_path).resolve())]
    cmd += [
        "-filter_complex", f"{vf};{af}",
        "-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", cfg.get("video.preset", "veryfast"),
        "-crf", str(cfg.get("video.crf", 21)), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart", str(out_path.resolve()),
    ]
    run(cmd, cwd=work)
    return out_path


# ------------------------------------------------------------------ streamer layout: webcam on top, gameplay below

def streamer_sizes(width: int, height: int, top_ratio: float) -> tuple[int, int]:
    """(height of the webcam zone, height of the gameplay zone); both even, together = height."""
    th = int(height * min(max(top_ratio, 0.2), 0.5)) // 2 * 2
    return th, height - th


def streamer_filter(cam: Box, crop: Box, width: int, height: int, fps: int, top_ratio: float,
                    ass_name: str | None) -> str:
    th, gz = streamer_sizes(width, height, top_ratio)
    cx, cy, cw, ch = (int(v) // 2 * 2 for v in cam)
    gx, gy, gw, gh = (int(v) // 2 * 2 for v in crop)
    bh = min(int(width * gh / gw), gz) // 2 * 2               # the gameplay block, never taller than its zone
    blur = "boxblur=26:3,eq=brightness=-0.18"
    graph = (
        f"[0:v]split=2[s1][s2];"
        f"[s1]crop={cw}:{ch}:{cx}:{cy},setsar=1,split[c1][c2];"
        f"[c1]scale={width}:{th}:force_original_aspect_ratio=increase,crop={width}:{th},{blur}[cbg];"
        f"[c2]scale={width}:{th}:force_original_aspect_ratio=decrease:flags=lanczos[cfg];"
        f"[cbg][cfg]overlay=(W-w)/2:(H-h)/2[top];"
        f"[s2]crop={gw}:{gh}:{gx}:{gy},setsar=1,split[g1][g2];"
        f"[g1]scale={width}:{gz}:force_original_aspect_ratio=increase,crop={width}:{gz},{blur}[gbg];"
        f"[g2]scale={width}:{bh}:flags=lanczos[gfg];"
        f"[gbg][gfg]overlay=0:(H-h)/2[bot];"
        f"[top][bot]vstack=inputs=2,fps={fps}"
    )
    if ass_name:
        graph += f",ass={ass_name}"
    return graph + "[v]"


def streamer_audio(cfg, dur: float, has_audio: bool) -> str:
    """The streamer's own voice and the game sound, exactly as recorded (only loudness is levelled)."""
    lufs = float(cfg.get("audio.target_lufs", -14))
    if has_audio:
        delay = float(cfg.get("streamer.audio_delay", 0) or 0)     # manual sync fix: > 0 = voice later, < 0 = voice earlier
        shift = ""
        if delay > 0:
            shift = f"adelay={int(round(delay * 1000))}:all=1,"
        elif delay < 0:
            shift = f"atrim=start={-delay:.3f},asetpts=PTS-STARTPTS,"
        return (f"[0:a]aresample=48000,{shift}atrim=0:{dur:.3f},afade=t=in:st=0:d=0.12,"
                f"afade=t=out:st={max(0.0, dur - 0.25):.3f}:d=0.25,loudnorm=I={lufs}:TP=-1.5:LRA=11[a]")
    return f"anullsrc=r=48000:cl=stereo,atrim=0:{dur:.3f}[a]"


def render_streamer(video: str | Path, start: float, dur: float, cam: Box, crop: Box, ass_path: Path | None,
                    out_path: Path, cfg, has_audio: bool = True) -> Path:
    out_path = Path(out_path)
    work = ass_path.parent if ass_path else out_path.parent
    width, height = int(cfg.get("video.width")), int(cfg.get("video.height"))
    vf = streamer_filter(cam, crop, width, height, int(cfg.get("video.fps")),
                         float(cfg.get("streamer.top_ratio", 0.36)), ass_path.name if ass_path else None)
    af = streamer_audio(cfg, dur, has_audio)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(Path(video).resolve()),
        "-filter_complex", f"{vf};{af}",
        "-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", cfg.get("video.preset", "veryfast"),
        "-crf", str(cfg.get("video.crf", 21)), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart", str(out_path.resolve()),
    ]
    run(cmd, cwd=work)
    return out_path
