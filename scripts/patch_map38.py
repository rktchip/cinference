"""MAP38: det-splitk spec-off long-prompt 8x8 CLOSED."""
import json

p = "/mnt/c/src/cinference/docs/forward-map.json"
d = json.load(open(p))
assert isinstance(d["steps"], list), type(d["steps"])
d["steps"].append({
    "id": "det_splitk_specoff_long_8x8",
    "status": "SEALED_2026_09_26",
    "off_8x": "2 texts (375/385 chars, 4/4 split, diverge char 127); "
              "reproduces earlier 5/6 flip exactly - short-prompt clean was luck",
    "on_8x": "8/8 IDENTICAL, pins the 385-char variant (member of off set)",
    "verdict": "fix removes the atomic race and pins one deterministic result; "
               "det-splitk on spec-off CLOSED",
    "artifacts": ["/root/det_specoffL0.json", "/root/det_specoffL1.json",
                  "/root/Soff.ref", "/root/Son.ref"],
})
json.dump(d, open(p, "w"), indent=1)
print("MAP38 sealed")
