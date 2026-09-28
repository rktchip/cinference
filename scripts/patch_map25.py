import json

p = "docs/forward-map.json"
fm = json.load(open(p))
steps = fm.setdefault("steps", [])

steps.append({
    "id": "step7_split_oracle",
    "status": "SEALED_2026_09_26",
    "exe": "6c792539c05ddec6f2b12f7aae6f3b9ec37ad71574cea5676ecbe173c9082738",
    "sessions": [
        {"cap": "natural", "reps": 9, "log": "/root/split_nat.log",
         "restore": "162/162 OK", "null": "0 OK / 162 VOID (atomic floor)",
         "accept_diff": 5},
        {"cap": 0, "reps": 3, "log": "/root/split_cap0.log",
         "restore": "98/98 OK", "accept_diff": 0, "id_max": 0.5156},
        {"cap": 3, "reps": 3, "log": "/root/split_cap3.log",
         "restore": "53/53 OK", "accept_diff": 2}
    ],
    "verdict": ("PASS under fallback interpretation: restore 313/313 exact; "
                "all 7 ACCEPT-DIFFs at tie-class mtp-gaps (0.0-4.9, three "
                "<=1.5); F=43/F=94 flip both directions across sessions; "
                "no clear-gap failure; no NaN; atomics hypothesis weakened, "
                "verify-vs-accept L0 attribution recorded not proven"),
    "stamps": ["/root/split0.ref/.end", "/root/splitcap0.ref/.end",
               "/root/splitcap3.ref/.end"],
    "next": "step8 8/8+drift on step4-overhead bundle, then A-B-B-A timing"
})
json.dump(fm, open(p, "w"), indent=1)
print("MAP25-OK steps=%d" % len(steps))
