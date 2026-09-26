import sys, json, time, hashlib, urllib.request, subprocess, threading
# Row-23 confirm: TTFT at two prompt lengths on the same boot.
# Slope ms/prompt-tok ~13 => mirrored target row (one graph replay per
# prompt token). Usage field gives exact prompt_tokens; no estimation.
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
port = sys.argv[1] if len(sys.argv) > 1 else "8902"
tag = sys.argv[2] if len(sys.argv) > 2 else "ttft"
exe_sha = sys.argv[3] if len(sys.argv) > 3 else "unstamped"
FOX = ("Write a short passage about a fox crossing a river at dawn. "
       "Describe what it sees and hears in three sentences.")
# P300: frozen 4x-length text (FOX theme, numbered sections, no randomness).
P300 = " ".join(
    ["Section %d. %s" % (i, FOX) for i in range(1, 9)])
PROMPTS = [("p75", FOX), ("p300", P300)]
HASHES = {n: hashlib.sha1(p.encode()).hexdigest()[:12] for n, p in PROMPTS}
print("%s PROMPTS %s EXE=%s" % (tag, " ".join(
    "%s=%s" % (n, HASHES[n]) for n, _ in PROMPTS), exe_sha), flush=True)
def post(prompt, stream, ntok=16):
    b = json.dumps({"model": "exl3:Qwen3.8-27B-EXL3-3.5bpw",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": ntok, "temperature": 0, "stream": stream,
        "enable_thinking": False}).encode()
    return urllib.request.Request(
        "http://127.0.0.1:%s/v1/chat/completions" % port,
        data=b, headers={"Content-Type": "application/json"})
ptoks = {}
for name, prompt in PROMPTS:
    # exact prompt_tokens, non-stream (one call)
    try:
        with urllib.request.urlopen(post(prompt, False), timeout=900) as f:
            m = json.load(f)
        ptoks[name] = m.get("usage", {}).get("prompt_tokens", -1)
    except Exception as e:
        print("%s %s usage FAILED: %s" % (tag, name, e), flush=True)
        ptoks[name] = -1
    for r in range(2):
        t0 = time.time()
        n = 0
        t_first = None
        try:
            with urllib.request.urlopen(post(prompt, True), timeout=900) as f:
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
        ttft = (t_first - t0) if t_first else dt
        print("%s %s[%s] rep %d: ptok=%s %d toks wall %.1fs ttft %.2fs decode %.2f ms/tok" % (
            tag, name, HASHES[name], r, ptoks[name], n, dt, ttft,
            (dt - ttft) / n * 1000 if n else -1), flush=True)
print("%s DONE" % tag, flush=True)
_load_stop = True
if _load_clocks:
    print("%s LOADCLK n=%d mean=%dMHz min=%d max=%d" % (tag, len(_load_clocks),
        sum(_load_clocks) // len(_load_clocks), min(_load_clocks),
        max(_load_clocks)), flush=True)
