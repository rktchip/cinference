"""MAP36: 8a verdict (slotsmtp_node node-trace, FOX-64 traced req, tid ...732, 19 steps).

H0 HOLDS (low end):
- verify: 16 full-layer ranges/step, span 16-24ms (median ~22), ~1200 kernels
  ~8ms kernel sum, graph=0 (all individual). GDN verify (45 ranges) overlaps span.
- trailing: ONE graph launch, 1922 kernels, 11.5-11.6ms every step (rock steady).
- drafts: mtp.forward 5-8/step ~2-3ms NVTX; gap individuals ~500-770 kern ~5.5-6.9ms.
- step ~51ms = verify ~22 + gap ~29 (graph 11.6 + indiv ~6 + overhead ~10-11).

D2H (item 4 settled): 11/step. Blocking: 1x16B accept (~0.8-1.6ms stall) +
3x draft tokens (~0.8-1.5ms each). 7 small (2-4B) pipelined 25-250us gaps.
Static ~5 vs design 2 reconciled: design missed the 3 draft readbacks.
Fold draft chaining into on-device accept (after step 9).

STEP 9 RE-PREDICT (honest): verify kernel content ~8ms, floor ~6.5 -> graphed
~9ms. Saves ~13/step (22->9). Step ~51->~38. FOX ~11.3 vs spec-off 13.2.
Kill bar 10/step CLEARED (margin ~3) [medium: M=4 capturability untested].
Verify-replay must cover GDN layers too (45 ranges inside span) - check probe branch.

SPLIT-IN-BINARY: 8b6ff35 ancestor of det-splitk HEAD. Exe 9e3847ebdc650675.
NEXT: S-number A-B-B-A (e28cacb vs this exe, slots MTP, ms/step) -> spec-off
256tok 8xoff/8xon -> step 9. Pipe after MTP (13.2->11.2-11.5 [medium]).
Concurrency only if 13a clears bar.
"""
import json

p = "docs/forward-map.json"
d = json.load(open(p))
d["map36_8a_verdict"] = {
    "h0": "HOLDS-low-end",
    "verify_span_ms": "16-24 (median ~22), 16 full-layer ranges, ~1200 kern ~8ms, all individual",
    "trailing": "ONE graph launch, 1922 kern, 11.5-11.6ms every step",
    "drafts": "mtp.forward 5-8/step ~2-3ms; gap individuals ~6ms",
    "step_ms": "~51 = verify ~22 + gap ~29 (graph 11.6 + indiv ~6 + overhead ~10-11)",
    "d2h_per_step": 11,
    "blocking_syncs": "16B accept 0.8-1.6ms + 3x draft tokens 0.8-1.5ms; 7 small pipelined",
    "step9": "verify 22->~9, saves ~13/step, step ~51->~38, FOX ~11.3 vs 13.2. Kill<10 CLEARED [medium]",
    "step9_caveat": "replay must cover GDN layers (45 ranges in span)",
    "split_in_binary": "8b6ff35 ancestor of HEAD; exe 9e3847ebdc650675",
    "trace": "/root/slotsmtp_node.nsys-rep + /root/slotsmtp.sqlite, drift 0%",
}
d["steps"].append({
    "id": "8a_mtp_slots_capture",
    "status": "SEALED_2026_09_26_H0_HOLDS_LOW_END",
    "h0": "HOLDS (low end)",
    "verify_span_ms": "16 full-layer ranges/step, 16-24 (median ~22), ~1200 kern ~8ms, graph=0 all-individual; GDN (45 ranges) overlaps span",
    "trailing": "ONE graph launch, 1922 kern, 11.5-11.6ms every step",
    "drafts": "mtp.forward 5-8/step ~2-3ms; gap individuals ~500-770 kern ~5.5-6.9ms",
    "step_ms": "~51 = verify ~22 + gap ~29 (graph 11.6 + indiv ~6 + overhead ~10-11)",
    "d2h_per_step": 11,
    "blocking_syncs": "16B accept 0.8-1.6ms + 3x draft tokens 0.8-1.5ms; 7 small pipelined 25-250us",
    "sync_reconciliation": "static ~5 vs design 2 settled: design missed 3 draft readbacks; fold chaining into on-device accept (after step 9)",
    "step9_repredict": "verify kernel ~8ms floor ~6.5 -> graphed ~9ms; saves ~13/step; step ~51->~38; FOX ~11.3 vs 13.2. Kill<10 CLEARED margin ~3 [medium: M=4 capturability untested]",
    "step9_caveat": "replay must cover GDN layers (45 ranges in span); check probe branch",
    "split_in_binary": "8b6ff35 ancestor of det-splitk HEAD; exe 9e3847ebdc650675",
    "trace": "/root/slotsmtp_node.nsys-rep + /root/slotsmtp.sqlite, drift 0%",
})
json.dump(d, open(p, "w"), indent=1)
print("MAP36 sealed")
