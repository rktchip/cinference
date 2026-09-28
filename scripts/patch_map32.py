import json

p = "docs/forward-map.json"
fm = json.load(open(p))
fm.setdefault("steps", []).append({
    "id": "upstream_verdict_drop",
    "status": "SEALED_2026_09_26_DROP",
    "measured": ("upstream 1.5.1 (.pyd, tabby cu132 torch): kv 21.5-22.1us "
                 "(92-123 GB/s), qk 21.5-22.5us (176-236 GB/s); ours: kv "
                 "12.6/11.9, qk 13.1/12.4-19.8"),
    "verdict": ("best speedup 0.61x -> DROP porting for good per 1.5x rule; "
                "upstream gemm~=gemv, no small-shape advantage; our "
                "small-shape excess is ours to fix via concurrency"),
    "env_note": ("stock venv ships torch+cpu so .pyd never loads there; "
                 "runs under tabby-exl3 venv (cu132) with cublas preload "
                 "wrapper scripts/run_upstream_bench.py (MSYS PATH entries "
                 "unusable by Win loader, preload by full path)")
})
json.dump(fm, open(p, "w"), indent=1)
print("MAP32-OK")
