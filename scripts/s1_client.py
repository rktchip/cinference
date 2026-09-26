import sys, json, time, urllib.request
# S1 decode arm: FOX prompt, greedy, conc-1, N output tokens, R reps.
# Prints per-rep toks + wall time; ms/tok = wall/toks (decode-dominated
# at conc-1; TTFT reported separately from the serve log req line).
# Usage: s1_client.py <port> <tag> [ntok=64] [reps=3]
port = sys.argv[1] if len(sys.argv) > 1 else "8902"
tag = sys.argv[2] if len(sys.argv) > 2 else "s1"
ntok = int(sys.argv[3]) if len(sys.argv) > 3 else 64
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
FOX = ("Write a short passage about a fox crossing a river at dawn. "
       "Describe what it sees and hears in three sentences.")
PARIS = ("Describe the city of Paris in three sentences: one about its "
         "history, one about its architecture, and one about its food.")
PROMPTS = [("fox", FOX), ("paris", PARIS)]
for name, prompt in PROMPTS:
    for r in range(reps):
        b = json.dumps({
            "model": "exl3:Qwen3.8-27B-EXL3-3.5bpw",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": ntok, "temperature": 0, "stream": True}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:%s/v1/chat/completions" % port,
            data=b, headers={"Content-Type": "application/json"})
        t0 = time.time()
        n = 0
        try:
            with urllib.request.urlopen(req, timeout=900) as f:
                for line in f:
                    s = line.decode(errors="replace")
                    if s.startswith("data:") and s[5:].strip() not in ("[DONE]", ""):
                        try:
                            m = json.loads(s[5:])
                        except Exception:
                            continue
                        if m["choices"][0]["delta"].get("content"):
                            n += 1
        except Exception as e:
            print("%s %s rep %d FAILED: %s" % (tag, name, r, e), flush=True)
            continue
        dt = time.time() - t0
        print("%s %s rep %d: %d toks in %.1fs = %.2f ms/tok" % (tag, name, r, n, dt, dt / n * 1000),
              flush=True)
print("%s DONE" % tag, flush=True)
