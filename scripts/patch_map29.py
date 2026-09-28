import json

p = "docs/forward-map.json"
fm = json.load(open(p))
fm.setdefault("steps", []).append({
    "id": "revised_step4_width_bench",
    "status": "SEALED_2026_09_26_AWAITS_CHIP_CALL",
    "paper": "Chip: 3-bit kernels at 4-bit speed -> 13.2 to ~11.4 (~1.8ms)",
    "measured": {
        "device": "RTX 5090 cc120 sms170 m1 cb2(mul1)",
        "note": "same shape, same cb; plan grid sometimes differs by "
                "width (gateup: 272 vs 340) as production does",
        "attn_q_5120x6144": "3-bit 18.34us/645GBs vs 4-bit 16.64/947 (+1.7)",
        "attn_kv_5120x1024": "12.64/157 vs 11.90/221 (+0.7)",
        "attn_o_6144x5120": "13.73/861 vs 13.22/1192 (+0.5)",
        "gdn_qk_5120x2048": "13.12/301 vs 12.42-19.78/266 (unstable, ~0)",
        "mlp_gateup_5120x17408": "25.44/1316 vs 27.68/1612 (-2.2, 3-bit FASTER)",
        "mlp_down_17408x5120": "27.14/1233 vs 23.84/1871 (+3.3)"
    },
    "verdict": {
        "us": "MIXED sign (-2.2..+3.3us), net ~+4us/layer ~+0.3ms/token",
        "per_byte": "3-bit GB/s lower almost everywhere (decode overhead "
                    "real) but fewer bytes wash it in absolute time",
        "capture_37v20": "does NOT replicate same-shape; was shape mix "
                         "(3-bit tensors are the big ones)",
        "prediction_miss": "1.8 predicted vs ~0.3 measured (>2x) -> "
                           "requant route does NOT pay (+1.5GB VRAM for "
                           "~0.3ms); tune-decode route same prize",
        "redirect": "excess is general small-shape efficiency "
                    "(kv/qk at 150-420 GB/s, 12-25% of peak) + "
                    "launch/grid overhead, not width"
    },
    "question_for_chip": ("kill width line (recommend), or bench upstream "
                          "exllamav3 coop kernel on these shapes first?"),
    "build_fix": ("bench assumed external exl3_gemv_run; it is __host__ "
                  "inline in exl3_launcher.cu -> de-inlined one word "
                  "(zero behavior change); isolated build dir "
                  "/root/cinference-bench-build, main build/ untouched"),
    "binary": "/root/cinference-bench-build/bench/ninfer_exl3_gemv_width_bench"
})
json.dump(fm, open(p, "w"), indent=1)
print("MAP29-OK")
