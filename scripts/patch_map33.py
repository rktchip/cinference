import json

p = "docs/forward-map.json"
fm = json.load(open(p))
fm.setdefault("steps", []).append({
    "id": "step9_timing_rule",
    "status": "SEALED_2026_09_26_RULE_SET",
    "rule": ("step-9 timing runs WITHOUT NINFER_EXL3_FORCE_SPLIT1 "
             "(split-1 carries the verify occupancy cliff: kv m4 8 vs "
             "~88 blocks; slots-vs-legacy must not compare a handicapped "
             "verify)"),
    "split1_gate": "verify-only tool (5x determinism proof), never for timing",
    "true_fix": ("deterministic split-K with occupancy preserved "
                 "(per-split partials + ordered reduce or equivalent), "
                 "background write-only; step 9 uses it only if built + "
                 "verified by then, else runs flag-free with noted "
                 "nondeterminism (tie-class, bimodal flips)")
})
json.dump(fm, open(p, "w"), indent=1)
print("MAP33-OK")
