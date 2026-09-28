import json

p = "docs/forward-map.json"
fm = json.load(open(p))
fm.setdefault("steps", []).append({
    "id": "step8_abba_bundle_timing",
    "status": "SEALED_2026_09_26_ESCALATED",
    "old_exe": "e28cacb",
    "new_exe": "9b9b54b7885894900cc991954139d1fb860b116f5e2095f21faef34e1af6ed41",
    "legs": [
        {"leg": "A1", "bin": "old", "median_ms_tok": 13.00,
         "range": "12.95-13.04", "loadclk": 2609},
        {"leg": "B1", "bin": "new", "median_ms_tok": 12.97,
         "range": "12.89-13.06", "loadclk": 2565},
        {"leg": "B2", "bin": "new", "median_ms_tok": 13.20,
         "range": "13.11-13.24", "loadclk": 2566},
        {"leg": "A2", "bin": "old", "median_ms_tok": 13.18,
         "range": "13.13-13.26", "loadclk": 2588}
    ],
    "mirrored_pairs": {"A1B1": -0.03, "B2A2": +0.02},
    "lifetime_drift": {"B1_to_B2": +0.23, "A1_to_A2": +0.18,
                       "note": "drift 6-7x larger than A/B delta; "
                               "temp 37-38C across session"},
    "verdict": ("bundle effect UNRESOLVABLE at noise floor; prediction "
                "-0.8..-1.2ms missed by >10x -> ESCALATED per >2x rule; "
                "kill bar (<0.3 gain) also fails; B1 end-VOID was ghost-lag "
                "transient (B2 pre GO shared_live=0); B0 pilot discarded "
                "(env mismatch, unified for A1/B1/B2/A2)"),
    "stamps": ["/root/abba_A1.ref/.end", "/root/abba_B1.ref/.end",
               "/root/abba_B2.ref/.end", "/root/abba_A2.ref/.end"],
    "next": "Chip decides: kill bundle, or re-probe with thermal controls"
})
json.dump(fm, open(p, "w"), indent=1)
print("MAP26-OK")
