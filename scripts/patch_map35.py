"""MAP35: MTP-first static (item 1, no GPU). Slots MTP step per phase.

Bonus: argmax + sync + blocking D2H (engine.cpp:1043).
Drafts x3: mtp_forward_ar_step + sync + blocking D2H each (:1077).
Verify M=4: EAGER in slots ("eager-slots", :1827; graph gated !slots_on
at :1779; slots call carries extra vcoltab_p so it cannot replay).
~710 launches by capture lore. + sync + blocking D2H host_targets (:1833).
Commit rows 0..len-2 SKIPPED in slots (copy-back path :1934+).
Trailing row: run_single_row(allow_graph=true) (:1966) — captures
commit_exec_ first call, REPLAYS after; no slots gate; fail-closed via
commit_dead_. NO sample/sync on trailing (token already known).
MTP refills: mtp_forward_batch x(commit_len-1), D2D staged.
Host syncs/step: ~5 blocking (bonus+3 drafts+verify); trailing sync-free.

H0 assessment: H0's trailing-EAGER half is contradicted by code —
trailing replays unless commit_dead_. If replay is live, H0's 28+28
split is wrong; likely split is verify-eager-M4 (~4x bytes: floor
~24ms + launch overhead) + trailing-replay (~13) + drafts/refills.
The 8a capture decides: count graph launches vs ~700-individual on
the trailing row, and measure verify-M4-eager directly.
"""
import json
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "docs" / "forward-map.json"
fm = json.loads(p.read_text())
steps = fm.setdefault("steps", [])
steps.append({"id": "mtp_first_static",
    "status": "SEALED",
    "verify_slots": "EAGER (eager-slots :1827, graph gated !slots_on :1779)",
    "trailing_row": "REPLAYS commit_exec_ (:1966 allow_graph=true, no slots gate)",
    "commit_rows": "SKIPPED in slots (copy-back :1934+)",
    "syncs_per_step": "5 blocking (bonus+3 drafts+verify); trailing sync-free",
    "H0": "trailing-eager half CONTRADICTED by code; 8a capture decides",
    "next": "8a node-traced slots MTP capture + det-splitk verify, same window",
})
p.write_text(json.dumps(fm, indent=1))
print("MAP35 sealed")
