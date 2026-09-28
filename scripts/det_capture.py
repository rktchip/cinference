import sys, json, time, urllib.request
# Determinism probe: 5 identical MTP runs, save full token streams for diff.
# Usage: det_capture.py <port> <tag> [ntok=64] [reps=5]
port = sys.argv[1] if len(sys.argv) > 1 else "8902"
tag = sys.argv[2] if len(sys.argv) > 2 else "det"
ntok = int(sys.argv[3]) if len(sys.argv) > 3 else 64
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 5
prompt = ("Write a short passage about a fox crossing a river at dawn. "
          "Describe what it sees and hears in three sentences.")
streams = []
for r in range(reps):
    b = json.dumps({
        "model": "exl3:Qwen3.8-27B-EXL3-3.5bpw",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": ntok, "temperature": 0, "stream": True,
        "enable_thinking": False}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%s/v1/chat/completions" % port,
        data=b, headers={"Content-Type": "application/json"})
    toks = []
    try:
        with urllib.request.urlopen(req, timeout=900) as f:
            for line in f:
                s = line.decode(errors="replace")
                if s.startswith("data:") and s[5:].strip() not in (
                        "[DONE]", ""):
                    try:
                        m = json.loads(s[5:])
                    except Exception:
                        continue
                    d = m["choices"][0].get("delta", {})
                    c = d.get("content")
                    if c:
                        toks.append(c)
    except Exception as e:
        print("%s rep %d FAILED: %s" % (tag, r, e), flush=True)
        continue
    streams.append("".join(toks))
    print("%s rep %d: %d chunks %d chars" % (tag, r, len(toks),
                                             len(streams[-1])), flush=True)
base = streams[0] if streams else ""
for r, s in enumerate(streams):
    i = next((i for i, (a, b) in enumerate(zip(base, s)) if a != b),
             min(len(base), len(s)))
    print("%s rep %d vs rep 0: common_prefix_chars=%d len=%d/%d %s" % (
        tag, r, i, len(s), len(base),
        "IDENTICAL" if s == base else "DIFFERS"), flush=True)
open("/root/det_%s.json" % tag, "w").write(json.dumps(streams))
print("%s DONE saved=/root/det_%s.json" % (tag, tag), flush=True)
