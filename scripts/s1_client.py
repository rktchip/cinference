import sys, json, time, hashlib, urllib.request, subprocess, threading
# S1 decode arm: frozen prompts (see s1_prompts.md), greedy, conc-1.
# Each S-line stamps the prompt sha1: names map to exactly one text.
# Load-clock sampler: mean SM clock under load prints at DONE; arm
# comparison uses load clocks (idle clocks bounce hundreds of MHz).
_load_clocks = []
_load_stop = False
def _sampler():
    while not _load_stop:
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=clocks.current.sm",
                "--format=csv,noheader,nounits"], capture_output=True,
                text=True, timeout=10).stdout.strip()
            _load_clocks.append(int(o.split()[0]))
        except Exception:
            pass
        time.sleep(2)
threading.Thread(target=_sampler, daemon=True).start()
# Usage: s1_client.py <port> <tag> [ntok=64] [reps=3]
port = sys.argv[1] if len(sys.argv) > 1 else "8902"
tag = sys.argv[2] if len(sys.argv) > 2 else "s1"
ntok = int(sys.argv[3]) if len(sys.argv) > 3 else 64
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
exe_sha = sys.argv[5] if len(sys.argv) > 5 else "unstamped"
PROMPTS = [
    ("fox", "Write a short passage about a fox crossing a river at dawn. "
     "Describe what it sees and hears in three sentences."),
    ("paris", "Paris is the capital of France. Explain why the Eiffel Tower "
     "was built, who designed it, and when the construction finished."),
    ("paris-city", "Describe the city of Paris in three sentences: one about "
     "its history, one about its architecture, and one about its food."),
]
HASHES = {n: hashlib.sha1(p.encode()).hexdigest()[:12] for n, p in PROMPTS}
print("%s PROMPTS %s EXE=%s" % (tag, " ".join("%s=%s" % (n, HASHES[n]) for n, _ in PROMPTS), exe_sha),
      flush=True)
for name, prompt in PROMPTS:
    for r in range(reps):
        b = json.dumps({
            "model": "exl3:Qwen3.8-27B-EXL3-3.5bpw",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": ntok, "temperature": 0, "stream": True,
            # Regime arms run thinking-OFF: thinking-on preambles draft
            # easily and flatten accept rates across prompts (2026-09-25).
            "enable_thinking": False}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:%s/v1/chat/completions" % port,
            data=b, headers={"Content-Type": "application/json"})
        t0 = time.time()
        n = 0
        t_first = None
        try:
            with urllib.request.urlopen(req, timeout=900) as f:
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
                            if t_first is None:
                                t_first = time.time()
        except Exception as e:
            print("%s %s rep %d FAILED: %s" % (tag, name, r, e), flush=True)
            continue
        dt = time.time() - t0
        if n == 0:
            print("%s %s rep %d: 0 toks, FAILED" % (tag, name, r), flush=True)
            continue
        ttft = (t_first - t0) if t_first else dt
        print("%s %s[%s] rep %d: %d toks wall %.1fs ttft %.1fs decode %.2f ms/tok" % (
            tag, name, HASHES[name], r, n, dt, ttft, (dt - ttft) / n * 1000), flush=True)
print("%s DONE" % tag, flush=True)
_load_stop = True
if _load_clocks:
    print("%s LOADCLK n=%d mean=%dMHz min=%d max=%d" % (tag, len(_load_clocks),
        sum(_load_clocks) // len(_load_clocks), min(_load_clocks),
        max(_load_clocks)), flush=True)
