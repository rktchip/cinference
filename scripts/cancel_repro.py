import json, urllib.request, sys
# Cancel repro: abandon one request mid-stream, then send a normal one.
port = sys.argv[1] if len(sys.argv) > 1 else "8902"
def req(tag, timeout, abort_after=None, ntok=40):
    b = json.dumps({
        "model": "exl3:Qwen3.8-27B-EXL3-3.5bpw",
        "messages": [{"role": "user",
                      "content": "Describe the history of the Eiffel Tower in detail."}],
        "max_tokens": ntok, "temperature": 0,
        "enable_thinking": False, "stream": True}).encode()
    r = urllib.request.Request(
        "http://127.0.0.1:%s/v1/chat/completions" % port,
        data=b, headers={"Content-Type": "application/json"})
    n = 0
    try:
        with urllib.request.urlopen(r, timeout=timeout) as f:
            for line in f:
                s = line.decode(errors="replace")
                if s.startswith("data:") and s[5:].strip() not in ("[DONE]", ""):
                    try:
                        m = json.loads(s[5:])
                    except Exception:
                        continue
                    if m["choices"][0]["delta"].get("content"):
                        n += 1
                        if abort_after is not None and n >= abort_after:
                            print("%s: abandoning after %d toks" % (tag, n),
                                  flush=True)
                            return "ABANDONED"
    except Exception as e:
        print("%s: ended after %d toks: %s" % (tag, n, e), flush=True)
        return "ENDED"
    print("%s: completed %d toks" % (tag, n), flush=True)
    return "OK"
print(req("cancel-1", 2, abort_after=2, ntok=200), flush=True)  # abandon in-flight
print(req("after-1", 600), flush=True)  # must survive the cancel
print("CANCELREPRO DONE", flush=True)
