"""MAP40: row-23 TTFT slope confirm + 8/8 matrix on ship binary a99b41e9."""
import json

p = "/mnt/c/src/cinference/docs/forward-map.json"
d = json.load(open(p))
assert isinstance(d["steps"], list), type(d["steps"])
d["steps"].append({
    "id": "R23_TTFT_slope_and_8x8",
    "status": "SEALED_2026_09_26",
    "ttft_probe": {
        "binary": "a99b41e93b033df5", "mode": "slots-MTP DET=1",
        "p75": {"sha": "5fb48df35619", "ptok": 35, "ttft_s": [1.09, 1.08]},
        "p300": {"sha": "caff2519b00f", "ptok": 228, "ttft_s": [6.47, 6.33]},
        "slope_ms_per_ptok": [27.9, 27.2],
        "intercept_s": 0.11,
        "verdict": "MIRROR CONFIRMED, UNIT COST 28 NOT 13: each prompt token "
                   "costs one EAGER target row (~28ms). Mirror calls "
                   "card_->ordinary_decode_batch directly (engine.cpp:2089), "
                   "no graph path. Intercept 0.11s ~= spec-off prefill. "
                   "Prediction (~13, graph-replay premise) MISSED by >2x -> "
                   "escalated; fix payoff is 2x larger than modeled.",
    },
    "matrix_8x8_ship_binary": {
        "legs": {"Rgate": "64x2", "R88": "64x3", "R256": "256x2",
                 "R64c": "64x1"},
        "self_identical": "8/8",
        "vs_E_detsplitk": {"short64": "== mtpdet reps (Rgate/R88/R64c)",
                           "long256": "== mtpdetL reps (R256, 402 chars)"},
        "verdict": "8/8 GREEN; R==E bit-exact incl 256-tok trajectory. "
                   "Correctness carried to ship binary.",
    },
    "static_read_rung_choice": {
        "mirror_site": "engine.cpp:2048 mtp_prefill_fill, per-row "
                       "ordinary_decode_batch + mtp_forward_batch",
        "rung1_caveat": "reused prefill hiddens come from batched-GEMM "
                        "prefill math, mirror rows use decode-GEMV math: "
                        "ULP-level different inputs to MTP KV. NOT bit-exact "
                        "by construction; gate must be accept-hists "
                        "(near-tie allowance), not token-identity.",
        "rung1b_option": "graph the mirror row (fixed width-1 shapes, same "
                        "math => bit-exact), ~28->~12ms/ptok; still fails "
                        "p2k parity (24s). Insufficient alone.",
        "rung2_api": "mtp_forward_batch accepts T in [1,prefill_chunk] "
                     "(text.cpp:631-634), batched M=N call is API-legal; "
                     "batched-vs-sequential MTP equivalence needs oracle "
                     "proof (bit-exact gate).",
        "recommendation": "rung-1 behind env flag first (measures the "
                         "hiddens question directly), then rung-2; skip "
                         "rung-1b except as fallback. Awaiting ack.",
    },
    "stamps": "pre R23.ref nproc=0; post SEALED (drift line = sm_clock "
              "idle->load swing only, idle-info-only; LOADCLK mean 2638MHz)",
})
json.dump(d, open(p, "w"), indent=1)
print("MAP40 sealed")
