import json

p = "docs/forward-map.json"
fm = json.load(open(p))
fm.setdefault("steps", []).append({
    "id": "step4_bundle_killed",
    "status": "SEALED_2026_09_26_KILLED_BY_CHIP",
    "branch": "step4-overhead (kept for reference, NOT merged)",
    "reason": ("A-B-B-A clean: -0.03/+0.02 vs predicted -0.8..-1.2; "
               "mirror cancels drift; prediction wrong not measurement; "
               "320 lines + 2 env flags not worth zero gain"),
    "why_missed": ("time is INSIDE linear kernels, not around them; "
                   "3-bit gemv 37us vs 4-bit 20us same grid: 25% fewer "
                   "bytes, 85% longer -> 3-bit trellis decode compute-bound"),
    "revised_step4": ("same-shape 3-bit vs 4-bit microbench (us + GB/s); "
                      "slower->tune decode OR speed-aware requant to 4-bit "
                      "(model-files change, back to Chip); "
                      "equal-per-byte->bench upstream coop kernel first"),
    "bucket2_idle": ("11.2 kernel vs 13.3 step = ~2.1ms GPU idle, new "
                     "bucket #2; test streaming-off, then host round-trip "
                     "-> launch next graph ahead of host work"),
    "determinism": ("5x MTP runs, diff streams, gap at first divergence; "
                    "split-K atomics suspect -> fixed-order M=4 reduction"),
    "reference": "spec-off ~13.0-13.2 fastest; relative bar moves with it",
    "unblocked_on": "e28cacb"
})
json.dump(fm, open(p, "w"), indent=1)
print("MAP27-OK")
