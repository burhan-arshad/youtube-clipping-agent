"""One ffmpeg call: cut -> crop to gameplay -> 9:16 -> captions + voice-over -> loudness -> mp4."""
from __future__ import annotations

from pathlib import Path

from .ffutil import run

Box = tuple[int, int, int, int]


def video_filter(crop: Box, mode: str, width: int, height: int, fps: int, ass_name: str | None) -> str:
    x, y, w, h = crop
    base = f"[0:v]crop={w}:{h}:{x}:{y},setsar=1"
    if mode == "hybrid":
        block_h = int(width * 5 / 4) // 2 * 2
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


def audio_filter(cfg, dur: float, has_game_audio: bool, lead_in: float) -> str:
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
    else:
        graph = f"{narr},"
    graph += f"atrim=0:{dur:.3f},loudnorm=I={lufs}:TP=-1.5:LRA=11[a]"
    return graph


def render(video: str | Path, start: float, dur: float, crop: Box, narration_wav: Path,
           ass_path: Path | None, out_path: Path, cfg, has_game_audio: bool = True) -> Path:
    """Render one Short. ffmpeg runs inside the clip folder so the ASS filename needs no escaping."""
    out_path = Path(out_path)
    work = ass_path.parent if ass_path else out_path.parent
    width, height = int(cfg.get("video.width")), int(cfg.get("video.height"))
    fps = int(cfg.get("video.fps"))
    vf = video_filter(crop, cfg.get("layout.mode", "fill"), width, height, fps,
                      ass_path.name if ass_path else None)
    af = audio_filter(cfg, dur, has_game_audio, float(cfg.get("narration.lead_in", 0.4)))
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(Path(video).resolve()),
        "-i", str(Path(narration_wav).resolve()),
        "-filter_complex", f"{vf};{af}",
        "-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", cfg.get("video.preset", "veryfast"),
        "-crf", str(cfg.get("video.crf", 21)), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart", str(out_path.resolve()),
    ]
    run(cmd, cwd=work)
    return out_path
