"""MAP44: two-reference seal -- drift 603/603, city 5-ULP record, 32k pair."""
import json

p = "C:/src/cinference/docs/forward-map.json"
d = json.load(open(p))
assert isinstance(d["steps"], list)
d["steps"].append({
    "id": "bisect_two_reference_seal",
    "status": "SEALED_2026_09_27",
    "rule": ("candidate token passes iff: equals spec-off argmax, OR within 2 ULP "
             "(scaled) of spec-off top, OR matches upstream exllamav3 greedy at "
             "that position. Fail requires disagreement with BOTH references beyond ties."),
    "bisect": {
        "legacy_MTP_e28cacb_slots_off": "FOX pauses (predates slots)",
        "presplit_slots_e28cacb_slots_on": "FOX pauses (not commit path)",
        "upstream_exllamav3_server_ids": "FOX pauses; CITY 682 (model truth, both prompts)",
        "forced_specoff_FOX_k2": "top 725 (curly 's) 17.5, pauses tied 3rd 17.375, 1 ULP TIE-CLASS",
        "specoff_eager_no_graph": "FOX 's (graph exonerated)",
        "specoff_e28cacb_Sep25": "FOX 's (predates row-20b/23)",
        "city_specoff_margin_k1": ("top 11 @20.375, 682 4th @19.75, 5 ULP -- outside tie "
                                    "allowance, passes via upstream condition. Recorded, not carved."),
    },
    "drift_longfox_603": ("forced-token scoring of full 603-token R2 output through "
                           "spec-off leg4 binary: 584 exact argmax + 19 tie-class "
                           "(worst exactly 2.0 ULP @276), 0 fails. "
                           "Prompt sha 933facf7fcd0, ids roundtrip-verified."),
    "p32k_pair": {
        "prompt": "32772 tokens (unit x1365 + template, HF-verified), kv 34816 conc 1 det 1",
        "specoff_verify_8a1c90c_warm_s": "11.34 (rep0 11.51)",
        "R2_rung2_3b7ff5d7_warm_s": "11.42 (rep0 11.54)",
        "overhead": "+80ms, 0.7%, inside 5% bar. Long context unlocked (measured, not projected).",
    },
    "consequence": ("No spec-off fix, no reference switch. Forced-token scoring is the "
                     "permanent gate. Spec-off handoff ticket (time-boxed, values) open."),
})
json.dump(d, open(p, "w"), indent=1)
print("MAP44 appended, steps:", len(d["steps"]))
