import threading
import time
from pathlib import Path
from types import SimpleNamespace

from featureclip import live
from featureclip.llm import LLMError
from featureclip.sources import Source


class FakeStore:
    def __init__(self):
        self.cand, self.judged = {}, {}

    def clips(self, vid, status):
        n = {"candidate": self.cand, "judged": self.judged}[status].get(vid, 0)
        return [object()] * n


def make_pipe(tmp_path, fail_first_n=0):
    st = FakeStore()
    calls = {"scan": 0, "judge": 0, "build": 0, "fail": fail_first_n}

    def scan(source_id, path=None):
        calls["scan"] += 1
        st.cand[calls["scan"]] = 2
        return calls["scan"]

    def judge(vid):
        calls["judge"] += 1
        if calls["fail"] > 0:                 # AI quota exhausted: nothing could be judged
            calls["fail"] -= 1
            raise LLMError("no usable model left")
        st.judged[vid], st.cand[vid] = 2, 0

    def build(vid):
        calls["build"] += 1
        st.judged[vid] = 0

    pipe = SimpleNamespace(sources={"s": Source(id="s", streamer="x")}, work=tmp_path, store=st, cfg=SimpleNamespace(get=lambda k, d=None: d),
                           scan=scan, judge=judge, build=build)
    return pipe, calls


def test_recorder_keeps_going_and_every_piece_is_processed(test_video, tmp_path):
    pipe, calls = make_pipe(tmp_path)
    t0 = time.time()
    n = live.watch(pipe, "s", local_test_file=str(test_video), segment_seconds=8, poll=0.5, ai_retry_seconds=1)
    assert n >= 3 and calls["scan"] == n and calls["build"] == n


def test_ai_limit_waits_and_retries_instead_of_losing_the_piece(test_video, tmp_path):
    pipe, calls = make_pipe(tmp_path, fail_first_n=2)
    live.watch(pipe, "s", local_test_file=str(test_video), segment_seconds=40, poll=0.5, ai_retry_seconds=1)
    assert calls["judge"] >= 3                # failed twice, then succeeded
    assert pipe.store.clips(1, "judged") == [] and pipe.store.clips(1, "candidate") == []


def test_stop_event_ends_the_run(test_video, tmp_path):
    pipe, calls = make_pipe(tmp_path)
    stop = threading.Event()
    threading.Timer(4, stop.set).start()
    start = time.time()
    live.watch(pipe, "s", local_test_file=str(test_video), segment_seconds=60, poll=0.5, stop=stop, ai_retry_seconds=1)
    assert time.time() - start < 40
    assert calls["scan"] >= 1                 # the partial last piece is still processed
