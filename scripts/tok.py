import sys, json
from tokenizers import Tokenizer
T = Tokenizer.from_file("C:/models/Qwen3.8-27B-EXL3-3.5bpw/tokenizer.json")
cmd = sys.argv[1]
if cmd == "count":
    print(len(T.encode(sys.argv[2]).ids))
elif cmd == "ids":
    print(json.dumps(T.encode(sys.argv[2]).ids))
elif cmd == "text":
    print(json.dumps(T.decode(json.loads(sys.argv[2]))))
elif cmd == "prefix-check":
    # verify text roundtrip: encode(prompt+output) and compare head
    ids = T.encode(sys.argv[2]).ids
    print(json.dumps({"n": len(ids), "head": ids[:8], "tail": ids[-8:]}))
