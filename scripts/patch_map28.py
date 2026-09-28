import json

p = "docs/forward-map.json"
fm = json.load(open(p))
fm.setdefault("steps", []).append({
    "id": "bucket2_nostream_and_determinism",
    "status": "SEALED_2026_09_26",
    "bucket2": {
        "nostream": "FOX 64: rep0 17.78 warmup, reps1-2 13.98/14.10 wall; "
                    "streaming wall 0.9s identical -> SSE costs ~0",
        "verdict": "2.1ms gap SURVIVES; host round-trip confirmed "
                   "(sample/readback/host/next-launch); fix: launch next "
                   "token graph ahead of host work, ~1-2ms/tok, helps all "
                   "MTP steps"
    },
    "determinism": {
        "design": "5x identical MTP FOX-64 on e28cacb (detA) + 5x on "
                  "split-bin with NINFER_MTP_LOGGAP=1 (detB)",
        "detA": "reps 0,1,3 identical; reps 2,4 diverge at char 83; "
                "rep2==rep4 (bimodal, single early flip)",
        "detB": "reps 0-3 identical text, rep4 diverges char 83; lanes "
                "alternate/req; within-lane commit reconstruction: rep2 "
                "flips tok23 279/514 render-identical (invisible), rep4 "
                "flips tok14 279/37728 (visible)",
        "gap": "flip row F=52 a=0: tok 37728 top 22.25 gap 0.125 "
               "(near-tie); good runs print a=2 row (430, gap ~1.4); "
               "draft-0 was 279 in ALL runs incl the flip",
        "trace": ("verify-side argmax flip on identical inputs; only "
                  "run-to-run nondeterminism in greedy single-GPU verify "
                  "GEMM is atomic split-K order "
                  "(torchexl3/exl3_gemm.cu:403,419-420); TRACED to "
                  "split-K reduction"),
        "fix": "fixed-order reduction at M=4 (cheap, per Chip conditional)",
        "effect": "no speed cost; restores exact-reproduction gates"
    }
})
json.dump(fm, open(p, "w"), indent=1)
print("MAP28-OK")
