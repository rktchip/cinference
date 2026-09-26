import json

P = "/mnt/c/src/cinference/docs/forward-map.json"
d = json.load(open(P))
e = d["engine_now"]

e["s1_pre_ABCCBA_2026_09_26"] = {
    "status": "SEALED medians of 6 reps/regime/mode, exe e71d9919c21d, "
              "all arms stamped",
    "ms_tok": {
        "slots": {"fox": 21.1, "paris": 15.8, "city": 15.7},
        "legacy": {"fox": 18.35, "paris": 17.6, "city": 17.65},
        "specoff": {"fox": 13.3, "paris": 13.3, "city": 13.3}},
    "ms_step": {"slots": "~63", "legacy": "~64", "specoff": "13.3"},
    "notes": "Second-half arms ~2-3% faster across modes (C1 13.43->C2 "
             "13.15, B1 fox 18.63->B2 18.04, A1 fox 21.27->A2 ~20.1); "
             "palindromic ABCCBA cancels drift to first order. Medians "
             "used: A2 fox rep2 23.86 outlier did not trip SEALED guard -> "
             "rep-level outlier flag TODO in harness. Trajmeans A2 hists "
             "misaligned by warmup groups, client ms/tok authoritative; "
             "tok/step slots ~3.8-4.3, legacy ~3.5."}

e["nondeterminism_2026_09_26"] = {
    "status": "OPEN finding, diagnostic proposed, no gate failed",
    "evidence": "Greedy token counts vary rep-to-rep same binary/mode: "
                "A2 fox 66/66/65 paris 64/64/67 city 67/64/67; B1 paris "
                "64/64/67 vs B2 67/67/67; A2 fox rep2 23.86 vs 20.1 median.",
    "proposal": "Determinism probe: same prompt x5 one server, compare "
                "streams; diverge-at-tie benign, diverge-at-clear-gap race "
                "bug hunt. 2-session box per plan rules.",
    "impact": "Bit-exact gates compare within one run (safe); histogram "
              "gates need distributions (safe); F-invariant assumed "
              "determinism (revisit if probe shows races)."}

e["step2_ABCCBA_2026_09_26"] = {"status": "LANDED", "arms": "A1 B1 C1 C2 B2 A2",
    "report": "See s1_pre_ABCCBA_2026_09_26."}

e["step3_bench_2026_09_26"] = {
    "status": "LANDED via capture attribution, synthetic bench parked unrun",
    "per_shape_real_kernel": {
        "gemv grid160 K4 20.4us x~81/tok, K3 37.0us x~44/tok": "K3 path "
            "1.8x slower per launch at same width; bulk 3.5bpw worst kernel",
        "gemv grid320 K4 23.5us x~47/tok, grid192 K4 20.1us x~47/tok": "ok",
        "gemm m3 50.6us SPLIT2, m4 59.8us SPLIT2, m4-grid112 26.7us SPLIT5": [
            "split-K on tiny grids, wave quantization + acc traffic",
            "SPLIT5 on 112-block grid = 560 blocks on 148-SM card"],
        "lm_head m5 30us x1/tok": "negligible"},
    "step13_gate": "PASS: small-m justified (50-60us/launch at ~0.3TB/s, "
                   "far over 1.3x floor)",
    "open_question": "m3/m4 gemm instances inside spec-off run need "
                     "prefill-vs-decode split by timestamp before step-4 "
                     "prediction (chunked-prefill hypothesis); floor recheck "
                     "from config in same script."}

json.dump(d, open(P, "w"), indent=1)
print("MAP19-OK")
