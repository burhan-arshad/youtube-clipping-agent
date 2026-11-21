"""Clean start and end for streamer clips: never cut a sentence in half.

The judge picks a window by time. A streamer clip is mostly speech, so a window that starts in the middle of a
sentence or ends before the punchline sounds broken. refine() moves the start back to the beginning of the sentence it
falls in and the end forward to the end of the sentence, using the word times from Whisper."""
from __future__ import annotations

Word = tuple[float, float, str]
ENDERS = (".", "?", "!", "…")


def _sentences(words: list[Word], gap: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    cur_s = cur_e = None
    prev_text = ""
    for s, e, w in sorted(words):
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s - cur_e >= gap or prev_text.rstrip().endswith(ENDERS):
            out.append((cur_s, cur_e))
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
        prev_text = w
    if cur_s is not None:
        out.append((cur_s, cur_e))
    return out


def refine(words: list[Word], start: float, end: float, *, video_len: float | None = None, gap: float = 0.45,
           lead: float = 0.35, tail: float = 0.5, max_back: float = 4.0, max_forward: float = 5.0,
           max_len: float | None = None) -> tuple[float, float, dict]:
    """Returns (new_start, new_end, info). Words are (start, end, text) in video time."""
    info = {"start_moved": 0.0, "end_moved": 0.0}
    sents = _sentences(words, gap)
    if not sents:
        return start, end, info
    ns, ne = start, end
    for s, e in sents:                                   # the start falls inside a sentence: go back to its beginning
        if s < start < e and start - s <= max_back and (start - s) > 0.05:
            ns = max(0.0, s - lead)
            break
    for s, e in sents:                                   # the end falls inside a sentence: finish it
        if s < end < e:
            if e - end <= max_forward:
                ne = e + tail
            else:                                        # too long to wait for: stop at the previous sentence end
                prev = [pe for ps, pe in sents if pe <= end and end - pe <= 3.0]
                if prev:
                    ne = max(prev) + tail
            break
    if video_len is not None:
        ne = min(ne, video_len)
    if max_len is not None and ne - ns > max_len:        # keep the limit: give back the extra at the end first
        prev = [pe for ps, pe in sents if pe <= ns + max_len - tail and pe > ns]
        ne = (max(prev) + tail) if prev else ns + max_len
    info["start_moved"] = round(start - ns, 2)
    info["end_moved"] = round(ne - end, 2)
    return ns, max(ne, ns + 1.0), info
