import sys, json, time, urllib.request
# Bucket-#2 probe: same S1 FOX prompt, stream=False. Compares wall ms/tok
# against the streaming arm to isolate SSE/streaming overhead from host
# round-trip idle. Usage: s1_nostream.py <port> <tag> [ntok=64] [reps=3]
port = sys.argv[1] if len(sys.argv) > 1 else "8902"
tag = sys.argv[2] if len(sys.argv) > 2 else "s1NS"
ntok = int(sys.argv[3]) if len(sys.argv) > 3 else 64
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
prompt = ("Write a short passage about a fox crossing a river at dawn. "
          "Describe what it sees and hears in three sentences.")
for r in range(reps):
    b = json.dumps({
        "model": "exl3:Qwen3.8-27B-EXL3-3.5bpw",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": ntok, "temperature": 0, "stream": False,
        "enable_thinking": False}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%s/v1/chat/completions" % port,
        data=b, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as f:
            m = json.load(f)
    except Exception as e:
        print("%s fox rep %d FAILED: %s" % (tag, r, e), flush=True)
        continue
    dt = time.time() - t0
    try:
        u = m.get("usage", {})
        n = u.get("completion_tokens") or len(
            m["choices"][0]["message"]["content"].split())
    except Exception:
        n = 0
    print("%s fox rep %d: %d toks wall %.2fs wall %.2f ms/tok" % (
        tag, r, n, dt, dt / max(n, 1) * 1000), flush=True)
print("%s DONE" % tag, flush=True)
