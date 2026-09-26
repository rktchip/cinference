"""MAP34: host-pipe verdict — PARK branch, concept validated on eager base.

Eager/Graph isolation (fix1 binary, FOX 64tok):
  graphed legacy (A1, 50a69f3 pipe-off): 13.14-13.22 ms/tok
  eager legacy (GRAPH=0, pipe off):       27.3 / 29.2 / 28.5 ms/tok
  eager+pipe (B1/B1b, PIPE=1, 6 reps):    25.5-27.7 ms/tok
Pipe saves ~2ms on the eager base (the ~2.1ms host idle is real and
removed) but the build runs the forward EAGER, giving up ~15ms of graph
replay savings. Net: 13ms regression vs graphed. KILL per <0.5 rule as
built. Fix-forward = capture pipe body into the graph (memcpy/event
nodes per lane); parked, not started. Concurrency is next per order #2
(it composes with graphs instead of fighting them).
Also: fixed pipe startup segfault on branch (afc77ef) — eligibility
dereferenced serve_sampling_ which is a DEVICE pointer; added host mirror.
Side finding: spec-off tie-flip is pre-existing (e28cacb 5/6 identical,
rep5 270-char variant, same char-127 split) — not a refactor artifact.
"""
import json
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "docs" / "forward-map.json"
fm = json.loads(p.read_text())
steps = fm.setdefault("steps", [])
steps.append({"id": "host_pipe_verdict",
    "status": "PARKED_KILLED_AS_BUILT",
    "branch": "host-pipe",
    "branch_tip": "afc77ef",
    "binaries": {
        "pipe_off_graphed_A1": {"sha": "50a69f3", "ms_tok": "13.14-13.22"},
        "eager_legacy_GRAPH0": {"sha": "fix1", "ms_tok": "27.3-29.2"},
        "eager_pipe_B1": {"sha": "7a35f785e2b8f4c8", "ms_tok": "25.5-27.7"},
    },
    "mechanism": "pipe removes ~2ms host idle on eager base but bypasses graph replay (~15ms)",
    "kill_rule": "saves <0.5ms vs graphed -> KILL as built",
    "fix_forward": "capture pipe body (forward+sample+D2D+D2H+event) into per-lane graph; parked",
    "bugfix": "eligibility deref of device serve_sampling_ -> host mirror (afc77ef)",
    "side_finding": "spec-off tie-flip pre-exists on e28cacb (5/6, char-127); exact gates need split-K + spec-off noise work",
    "next": "small-shape concurrency inside graphed body (order #2)",
})
fm["next_20"] = fm.get("next_20", 21)
p.write_text(json.dumps(fm, indent=1))
print("MAP34 sealed")
