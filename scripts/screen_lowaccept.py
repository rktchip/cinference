import sys, json, time, urllib.request
# Low-accept screen: 4 hostile candidates, LONG runs (~100 steps each,
# SE~0.1). Short screens (32 toks, ~11 steps, SE~0.3) cannot rank
# candidates — screen1-3 retired as unevidenced. Winner confirmed by
# rerun before freezing (minimum-of-noisy picks for luck).
# NOTE 2026-09-25: the "thinking flattens accepts" claim was struck —
# screen2-rand thinking-off 1.14 sits ABOVE screen1 thinking-on 1.00,
# opposite to the prediction, and all short-screen gaps are inside SE.
# Thinking-off stands on historical-gate grounds (matches the gate
# conditions), not on screen evidence.
port = sys.argv[1] if len(sys.argv) > 1 else "8902"
tag = sys.argv[2] if len(sys.argv) > 2 else "screen"
ntok = int(sys.argv[3]) if len(sys.argv) > 3 else 300
CANDS = [
    ("rand-nums", "Continue this sequence of random integers with the next "
     "five, separated by commas: 847293, 510382, 774916, 203957, 619048,"),
    ("nouns", "For each of these unrelated proper nouns, name an object "
     "it has nothing to do with: Xylophone, Garrulous, Quimper, Zzyzx."),
    ("invented", "Invent three words that have never existed in any "
     "language, each with a one-line definition."),
    ("coin", "Continue this exact coin-flip record with ten more flips, "
     "using only H and T separated by spaces: H T T H H T H T T H H T,"),
]
for name, prompt in CANDS:
    b = json.dumps({
        "model": "exl3:Qwen3.8-27B-EXL3-3.5bpw",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": ntok, "temperature": 0, "stream": True,
        "enable_thinking": False}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%s/v1/chat/completions" % port,
        data=b, headers={"Content-Type": "application/json"})
    t0 = time.time()
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as f:
            for line in f:
                s = line.decode(errors="replace")
                if s.startswith("data:") and s[5:].strip() not in ("[DONE]", ""):
                    try:
                        m = json.loads(s[5:])
                    except Exception:
                        continue
                    d = m["choices"][0].get("delta", {})
                    if d.get("content") or d.get("reasoning_content"):
                        n += 1
    except Exception as e:
        print("%s %s FAILED: %s" % (tag, name, e), flush=True)
        continue
    print("%s %s: %d toks in %.1fs" % (tag, name, n, time.time() - t0), flush=True)
print("%s DONE" % tag, flush=True)
