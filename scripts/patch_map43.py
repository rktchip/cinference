"""MAP43: rung-2 seal -- slope, p2k overhead, identity, hists."""
import json

p = "/mnt/c/src/cinference/docs/forward-map.json"
d = json.load(open(p))
assert isinstance(d["steps"], list), type(d["steps"])
d["steps"].append({
    "id": "R23_rung2_seal",
    "status": "SEALED_2026_09_26",
    "binary": "rung2 3b7ff5d7 (row23-rung1 commit a3153bc; NINFER_MTP_WARM_HIDDENS=1 NINFER_MTP_WARM_BATCHED=1)",
    "slope": {
        "p35_ttft_s": "0.08-0.09 (R2/R2b probe, clean window)",
        "p228_ttft_s": "0.15",
        "slope_ms_per_ptok": "0.31, target-prefill-dominated (mirror 28, v1 1.27)",
        "specoff_p35_s": "0.06-0.25 (rep1 0.06, rep0 warmup 0.25) -- R2 at parity",
    },
    "p2k_overhead": {
        "prompt": "~1.2k tokens (unit x45 + template; 2 chunks at prefill_chunk 1024, same as true 2k)",
        "R2_ttft_s": "0.41/0.42",
        "specoff_ttft_s": "0.39/0.38 (solo verify-8a1c90c, ss-verified, same prompt)",
        "overhead_ms": "+20 to +40, within pre-reg <=50ms [medium] PASS (near top of bar, honest read)",
        "p32k_projection": "~0.5s (32 chunks) vs multi-second spec-off prefill, few % -- unlocks long context (projection, not measured)",
    },
    "identity_vs_specoff": {
        "fox_det": "R2det bit-exact vs all 5 E streams (300ch)",
        "city_tie_prompt": "R2 == V1 char-for-char (682 at token 1, adjudicated 1-ULP tie, MAP42); no new divergence",
        "s1_fox_paris_texts": "R2 == V1 full 64-tok texts",
        "other_divergences": "none observed; any future one gets its own gap check per order",
    },
    "accepts": {
        "paired_fox": "19 steps, mean-a 1.474 both, hists identical {0:5,1:2,2:10,3:2}, zero per-step draft/accept diffs",
        "paired_paris": "17 steps, mean-a 1.941 both, hists identical {0:3,1:2,2:5,3:7}, zero diffs",
        "verdict": "rung-2 == v1 exactly on 36 steps; batched-vs-sequential MTP math moves no draft argmax here",
    },
    "windows": "R2 slope window pre-GO (LOADCLK 2845MHz under s1 load); R2c p2k window pre/post sealed drift 3% (R2b.ref)",
    "next": "FOX-500 drift on this final path, then on-device accept",
})
json.dump(d, open(p, "w"), indent=1)
print("MAP43 sealed")
