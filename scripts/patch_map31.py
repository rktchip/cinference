import json

p = "docs/forward-map.json"
fm = json.load(open(p))
fm.setdefault("steps", []).append({
    "id": "hostpipe_upstream_received",
    "status": "SEALED_2026_09_26_PLAN_SET",
    "host_pipe": {
        "branch": "host-pipe 50a69f3 (+283/-14, env NINFER_HOST_PIPE=1, default off)",
        "mechanism": ("device argmax -> persistent outbox -> async D2D to "
                      "next ids; host reads via pinned double mailbox + "
                      "event, streams one behind; drain drops finish-tail"),
        "caveat": ("agent never compiled (tried native-Windows configure, "
                   "claimed WSL2 absent - wrong); I build in WSL isolated "
                   "dir before any GPU run"),
        "test": "same-binary A-B-B-A pipe off/on vs 13.2->11.3-11.8 [med], kill <0.5"
    },
    "upstream": {
        "harness": "C:\\Users\\chipw\\upstream_gemv_harness\\upstream_exl3_gemv_bench.py (stock venv .pyd, kv+qk shapes)",
        "caveat": ("stock tree may already carry QTIP-style small-m GEMV "
                   "(460 vs 368 lines) - attribution unconfirmed; bias: "
                   "upstream incl had+scales, ours trellis-only"),
        "rule": ">=1.5x -> port that path, else drop porting for good"
    },
    "order_now": ["1. run upstream harness (minutes)", "2. build host-pipe isolated", "3. host-pipe A-B-B-A", "4. M=4 server build + 5x det verify", "5. MTP step 9"]
})
json.dump(fm, open(p, "w"), indent=1)
print("MAP31-OK")
