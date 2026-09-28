import json

p = "docs/forward-map.json"
fm = json.load(open(p))
fm.setdefault("steps", []).append({
    "id": "width_killed_orders",
    "status": "SEALED_2026_09_26_ORDERS_OPEN",
    "width": "KILLED by Chip: 37-vs-20 was shape mix, same-shape net "
             "~+0.3ms not worth 1.5GB VRAM or a re-quant",
    "next_build": ("host round trip (only measured bucket with a clear "
                   "fix): device argmax, token straight to next input, "
                   "launch next graph before host reads, host streams one "
                   "behind; predict 13.2->11.3-11.8 [med]; kill if <0.5; "
                   "feeds MTP sync reduction later"),
    "upstream_bench": ("yes, time-boxed, small shapes only (kv, qk); bar: "
                       "upstream >=1.5x there -> port that path, else drop "
                       "porting for good"),
    "linears_rank": ("per-shape excess BEFORE building concurrency fix; "
                     "small shapes -> parallel branches or one fused "
                     "launch; ~1ms of +4 [low]"),
    "m4_reduction": "fixed-order M=4 reduction in code lane NOW (small, restores exact-repro before MTP steps 9-11)",
    "order": ["1. host pipe build + upstream bench", "2. small-shape concurrency/fusion from ranked table", "3. MTP from step 9"]
})
json.dump(fm, open(p, "w"), indent=1)
print("MAP30-OK")
