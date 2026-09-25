import json, urllib.request, time, sys
# Accept-profile driver: sequential short greedy decodes against a port.
# Usage: sweep_client.py PORT TAG [NTOK]
port = sys.argv[1] if len(sys.argv) > 1 else "8902"
tag = sys.argv[2] if len(sys.argv) > 2 else "run"
ntok = int(sys.argv[3]) if len(sys.argv) > 3 else 40
prompts = [
    # F56 discriminator (prepend shifts absolute F): a split that stays at
    # F56 is position-keyed; one that moves with content is content-driven.
    "Yes. Write a short passage about a fox crossing a river at dawn. "
    "Describe what it sees and hears in three sentences.",
    "Paris is the capital of France. Explain why the Eiffel Tower was "
    "built, who designed it, and when the construction finished.",
    "Write a Python function that reverses a linked list in place, "
    "with a brief comment on each step of the algorithm.",
]
for i, p in enumerate(prompts):
    b = json.dumps({
        "model": "exl3:Qwen3.8-27B-EXL3-3.5bpw",
        "messages": [{"role": "user", "content": p}],
        "max_tokens": ntok, "temperature": 0,
        "enable_thinking": False, "stream": True}).encode()
    r = urllib.request.Request(
        "http://127.0.0.1:%s/v1/chat/completions" % port,
        data=b, headers={"Content-Type": "application/json"})
    t0 = time.time()
    n = 0
    try:
        with urllib.request.urlopen(r, timeout=600) as f:
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
        print("prompt %d FAILED: %s" % (i, e), flush=True)
        continue
    print("%s prompt %d: %d toks in %.1fs" % (tag, i, n, time.time() - t0),
          flush=True)
print("%s DONE" % tag, flush=True)
