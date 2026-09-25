import json, urllib.request, time, sys
# Accept-profile driver: sequential short greedy decodes against a port.
# Usage: sweep_client.py PORT TAG [NTOK]
port = sys.argv[1] if len(sys.argv) > 1 else "8902"
tag = sys.argv[2] if len(sys.argv) > 2 else "run"
ntok = int(sys.argv[3]) if len(sys.argv) > 3 else 40
# 4th arg: "prepend" adds the F56 position-test prefix (must stay 0 mod 3
# tokens — verify from the req prompt count); "long" appends the
# >64-token prefill control (starves mid-sequence page allocation below
# F120; assert max F < 120 from the verdict lines after the run).
mode = sys.argv[4] if len(sys.argv) > 4 else ""
fox = ("Write a short passage about a fox crossing a river at dawn. "
       "Describe what it sees and hears in three sentences.")
if mode == "prepend":
    fox = "Yes indeed. " + fox
prompts = [
    fox,
    "Paris is the capital of France. Explain why the Eiffel Tower was "
    "built, who designed it, and when the construction finished.",
    "Write a Python function that reverses a linked list in place, "
    "with a brief comment on each step of the algorithm.",
]
if mode == "long":
    # Prefill control: ~70-token prefill maps pages 0-1 at admission, so no
    # mid-sequence allocation fires below F120. Keep ntok small (<=27) so
    # max F stays < 120; assert from verdict F values after the run.
    prompts.append(
        "Summarize the plot of Pride and Prejudice in about sixty words, "
        "naming the two main characters and the central misunderstanding "
        "between them. Then add one sentence saying whether you recommend "
        "it to a first-time reader of classic novels and why.")
for i, p in enumerate(prompts):
    if mode == "long" and i < 3:
        continue
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
    texts = []
    try:
        with urllib.request.urlopen(r, timeout=600) as f:
            for line in f:
                s = line.decode(errors="replace")
                if s.startswith("data:") and s[5:].strip() not in ("[DONE]", ""):
                    try:
                        m = json.loads(s[5:])
                    except Exception:
                        continue
                    c = m["choices"][0]["delta"].get("content")
                    if c:
                        n += 1
                        texts.append(c)
    except Exception as e:
        print("prompt %d FAILED: %s" % (i, e), flush=True)
        continue
    with open("/root/sweep_%s_p%d.txt" % (tag, i), "w") as f:
        f.write("".join(texts))
    print("%s prompt %d: %d toks in %.1fs" % (tag, i, n, time.time() - t0),
          flush=True)
print("%s DONE" % tag, flush=True)
