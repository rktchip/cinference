"""MAP41: rung-1 v1 verdict + v2 kill + rung-2 reconcile."""
import json

p = "/mnt/c/src/cinference/docs/forward-map.json"
d = json.load(open(p))
assert isinstance(d["steps"], list), type(d["steps"])
d["steps"].append({
    "id": "R23_rung1_verdict",
    "status": "SEALED_2026_09_26",
    "binary": "fa241935f55c0e5c (row23-rung1 commit 20af6a1; rebuild reproduces hash)",
    "legs": "W0 flag-OFF (==E bit-exact, slope ~28) / W1 flag-ON",
    "rung1_measured": {
        "slope_ms_per_ptok": 1.27,
        "prediction": "0.3-0.6 MISSED ~2-4x; residual = eager launch tax on "
                      "sequential MTP forwards (~1ms/row). Rung-2 deletes it.",
        "p35_ttft_s": "0.13-0.14, near parity with spec-off 0.1-0.3",
        "fox_paris": "bit-exact vs mirror/E (64tok det + s1 regime)",
        "city": "token-0 bonus flip (prefill-math vs decode-math tail "
                "projection); isolated to the single rung-1-affected target "
                "value; deterministic stable alternate; accept 1.94->1.82. "
                "Classified tie-class: same distribution, two summation "
                "orders. Gate passes with allowance.",
        "hists": "FOX/paris byte-identical; city shape-similar, mean-a -6%",
        "no_mtp_logits": "fill calls mtp_forward_batch with logits_column=-1; "
                         "per-row lm_head reads already zero. Gap is launches.",
    },
    "v2_killed": {
        "idea": "tail row via ordinary path for decode-math bonus",
        "why_dead": "mirror rows chain GDN state shadow->shadow; skipping "
                    "non-tail rows leaves the shadow pristine, so the tail "
                    "read wrong GDN state. Measured: template-leak prefix "
                    "(system//The...), drafts/verify fully diverged. "
                    "Decode-math tail needs the full chain = full cost. "
                    "Binary 1538e5ad removed.",
    },
    "rung2_reconcile": {
        "correction": "MTP block has NO GDN state (stem+tail only; tail reads "
                      "batch_mtp_kv_). Shadow chaining belongs to the ordinary "
                      "rows skipped by rung-1/2. The width-N GDN risk is "
                      "withdrawn -- audit was right.",
        "corrected_risk": "span-N append+attend on the singular attention "
                          "path (per-row loop, span T): each row must append "
                          "its own K/V before observed. Empirical gate "
                          "(target-unchanged + hists) decides; no oracle.",
        "tail": "rung-2 inherits v1-style tail (prefill projection); same "
                "tie-class gate.",
    },
    "stamps": "W0/W1/W1p/Coff pre+post sealed; GPU idle, no paging (1195MiB)",
})
json.dump(d, open(p, "w"), indent=1)
print("MAP41 sealed")
