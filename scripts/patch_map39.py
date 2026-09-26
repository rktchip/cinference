"""MAP39: S9 seal (verify-replay E-R-R-E + E3 + same-session spec-off)."""
import json

p = "/mnt/c/src/cinference/docs/forward-map.json"
d = json.load(open(p))
assert isinstance(d["steps"], list), type(d["steps"])
legs = {
    "E1": {"bin": "detsplitk", "sha": "9e3847ebdc650675", "steps": 106,
           "decode_s": 5.5, "ms_per_step": 51.9,
           "regimes": {"fox": [15.57, 15.19], "paris": [12.81, 12.94], "city": [13.20, 13.88]},
           "loadclk": 2486},
    "R1": {"bin": "verify-8a1c90c", "sha": "a99b41e93b033df5", "steps": 106,
           "decode_s": 3.8, "ms_per_step": 35.85,
           "regimes": {"fox": [10.10, 10.17], "paris": [8.82, 9.00], "city": [8.89, 9.00]},
           "loadclk": 2326},
    "R2": {"bin": "verify-8a1c90c", "sha": "a99b41e93b033df5", "steps": 106,
           "decode_s": 3.8, "ms_per_step": 35.85,
           "regimes": {"fox": [10.35, 10.37], "paris": [9.09, 9.09], "city": [9.00, 9.08]},
           "loadclk": 2427},
    "E2": {"bin": "detsplitk", "sha": "9e3847ebdc650675", "steps": 106,
           "decode_s": 5.7, "ms_per_step": 53.8,
           "regimes": {"fox": [15.60, 16.13], "paris": [13.49, 13.25], "city": [13.21, 13.21]},
           "loadclk": 2485},
    "E3": {"bin": "detsplitk", "sha": "9e3847ebdc650675", "steps": 106,
           "decode_s": 5.5, "ms_per_step": 51.9,
           "regimes": {"fox": [15.95, 14.11], "paris": [13.92, 12.37], "city": [12.70, 13.90]},
           "loadclk": 2485},
}
E = 51.9  # median; E1/E3 exact agreement, E2 one-off noise +3.7%
R = 35.85
fox_R = (10.10 + 10.17 + 10.35 + 10.37) / 4
paris_R = (8.82 + 9.00 + 9.09 + 9.09) / 4
city_R = (8.89 + 9.00 + 9.00 + 9.08) / 4
avg_R = (fox_R + paris_R + city_R) / 3
off = [13.28, 13.15, 13.17, 13.25, 13.15, 13.16]
avg_off = sum(off) / len(off)
d["steps"].append({
    "id": "S9_verify_replay",
    "status": "SEALED_2026_09_26",
    "gate": "bit-exact R==E char-for-char (Rgate vs mtpdet); replay live (36 replay-slots)",
    "legs": legs,
    "mirrored_ms_per_step": {"E": E, "R": R, "delta": round(R - E, 2),
                             "pct": round(100 * (R - E) / E, 1)},
    "prediction_check": "predicted ~38; measured 35.85 (BEATS by ~2); kill bar -10 cleared by 6",
    "same_binary": "R1/R2 0.0%%; E1/E3 0.0%%; E2 +3.7%% one-off noise (uniform, clocks steady)",
    "regime_avg_ms_per_tok": {"MTP_R": round(avg_R, 2), "specoff_same_session": round(avg_off, 2),
                              "delta_pct": round(100 * (avg_R - avg_off) / avg_off, 1)},
    "fox_only": {"MTP_R": round(fox_R, 2), "specoff": 13.22},
    "relative_bar_15pct": "CLEARED (-28.7%% avg, -22.5%% FOX)",
    "open_flag": "MTP TTFT 1.0-1.2s vs spec-off 0.1-0.3s - TTFT parity NOT met, investigate",
    "stamps": "pre all legs nproc=0; post Soff9 SEALED drift=0%%",
})
json.dump(d, open(p, "w"), indent=1)
print("S9: E=%.1f R=%.2f d=%.2f; avg %.2f vs off %.2f (%.1f%%)" % (E, R, R - E, avg_R, avg_off, 100 * (avg_R - avg_off) / avg_off))
