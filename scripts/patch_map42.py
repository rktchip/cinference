"""MAP42: tie-class verdict (split skipped) + clean spec-off + overlap window."""
import json

p = "/mnt/c/src/cinference/docs/forward-map.json"
d = json.load(open(p))
assert isinstance(d["steps"], list), type(d["steps"])
d["steps"].append({
    "id": "R23_rung1_gate",
    "status": "SEALED_2026_09_26",
    "binary": "rung2 3b7ff5d7 (row23-rung1 commit a3153bc; flags gate v1/v2 paths)",
    "token0": "bit-identical 57590, gap 6.0 ULP both paths (G0/G1, pre-overlap)",
    "token1_row0_gaps": {
        "mirror_L0": "tok 11 top 20.0 gap 0.0 (~0 ULP, exact tie); solo-attributed by MTP-only trace",
        "v1_L1": "tok 682 top 20.12 gap 0.125 (= 1 ULP bf16@[16,32)); solo server, ss-verified",
        "verdict": "TIE-CLASS both sides inside 2-ULP margin; tops agree 20.0/20.12. "
                   "v1 did not overturn a decision, it called a tie the other way.",
    },
    "split": "SKIPPED per 2026-09-26 order: buys only step 1 while M=4-vs-M=1 verify "
             "rounding can flip later ties; gates stay tie-adjudicated; saves ~13ms TTFT + code.",
    "specoff_baseline": {
        "Soff10": "overlap-era (silent log, unattributable) -- DISCARDED as baseline",
        "Soff11": "clean rerun, solo verify-8a1c90c ss-verified, city 12tok: "
                  "'Paris, known as the \"City of Light,\" has served' (token1=11). "
                  "Matches Soff10 prefix. Conclusion stands: v1 is the odd one out, tie-class.",
    },
    "overlap_window": {
        "window": "Soff10 boot 16:03:48 -> full-cleanup kill ~17:21 (PDT 2026-09-26)",
        "cause": "pkill by binary name (rung2 pattern) missed verify-8a1c90c; two servers co-held :8902",
        "predates_outside": "S9/MAP39 12:45, R23+8/8/MAP40 13:00, MAP41 14:20, W0/W1, G0/G1 bonus gaps (15:46/15:58) -- all clean",
        "inside": "L0-att2 row-0 gap (trace-attributed, stands) + Soff10 city (discarded, rerun as Soff11)",
        "guard": "commit 7e0afa1: pre REFUSEs on held :8902 (ss holders shown); "
                 "teardown kills by port (INT, wait, verify, KILL fallback); verified live (TEARDOWN -INT 442, PORT-FREE)",
    },
    "rung2_gate": "identity vs spec-off (not v1/mirror) + accept hists within noise + TTFT slope; R2 booting",
})
json.dump(d, open(p, "w"), indent=1)
print("MAP42 sealed")
