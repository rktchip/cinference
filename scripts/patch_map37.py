"""MAP37: S_split ABBA seal (slots-MTP, pre-split e28cacb vs detsplitk DET=0)."""
import json

p = "/mnt/c/src/cinference/docs/forward-map.json"
d = json.load(open(p))
assert isinstance(d["steps"], list), type(d["steps"])
legs = {
    "A1": {"bin": "/root/ninfer-serve-e28cacb", "sha": "e71d9919c21d0e1c",
           "steps": 109, "decode_s": 7.0, "ms_per_step": 64.2,
           "fox": [22.12, 20.06], "loadclk": 2497},
    "B1": {"bin": "/root/ninfer-serve-detsplitk-528b29e", "sha": "9e3847ebdc650675",
           "steps": 106, "decode_s": 5.5, "ms_per_step": 51.9,
           "fox": [14.94, 17.15], "loadclk": 2419},
    "B2": {"bin": "/root/ninfer-serve-detsplitk-528b29e", "sha": "9e3847ebdc650675",
           "steps": 105, "decode_s": 5.6, "ms_per_step": 53.3,
           "fox": [14.61, 14.19], "loadclk": 2421},
    "A2": {"bin": "/root/ninfer-serve-e28cacb", "sha": "e71d9919c21d0e1c",
           "steps": 103, "decode_s": 7.3, "ms_per_step": 70.9,
           "fox": [19.63, 20.56], "loadclk": 2498},
}
A = (legs["A1"]["ms_per_step"] + legs["A2"]["ms_per_step"]) / 2
B = (legs["B1"]["ms_per_step"] + legs["B2"]["ms_per_step"]) / 2
foxA = sum(legs["A1"]["fox"] + legs["A2"]["fox"]) / 4
foxB = sum(legs["B1"]["fox"] + legs["B2"]["fox"]) / 4
d["steps"].append({
    "id": "S_split_ABBA",
    "status": "SEALED_2026_09_26",
    "metric": "ms/step = sum(wall-ttft)/serve-mtp-line-delta, script 3 regimes x2 reps",
    "legs": legs,
    "mirrored_ms_per_step": {"A": round(A, 1), "B": round(B, 1),
                             "delta": round(B - A, 2), "pct": round(100 * (B - A) / A, 1)},
    "fox_only_ms_per_tok": {"A": round(foxA, 2), "B": round(foxB, 2)},
    "prediction_check": "predicted 62.5 -> ~45 (save ~17.5); measured 67.6 -> 52.6 (save ~15.0)",
    "stamps": "pre all 4 legs nproc=0 no-spill; post live A2 SEALED drift=0%%; "
              "A1/B1/B2 posts lost (bad post chain + servers down) - documented",
    "identity": "explicit sha256 per binary file + launcher paths; "
                "preflight HASH tracks pinned exe only, NOT leg binding",
    "caveat": "B binary delta = split-K + dormant det scaffolding (flag off); "
              "attribution to split is conservative",
})
json.dump(d, open(p, "w"), indent=1)
print("S_split: A=%.1f B=%.1f d=%.2f (%.1f%%); FOX A=%.2f B=%.2f" % (A, B, B - A, 100 * (B - A) / A, foxA, foxB))
