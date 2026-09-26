import sqlite3
from collections import defaultdict

DB = "/root/speccoff_node.sqlite"
con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)

names = {}
for i, v in con.execute("SELECT id, value FROM StringIds"):
    names[i] = v

# bucket patterns: (label, [substrings])
PATS = [
    ("linear-exl3", ["exl3", "quant_", "q4", "gemm", "gemv", "cutlass", "nvfp4"]),
    ("gdn-recurrent", ["recurrent", "gated_delta", "delta_net", "gdn", "retention"]),
    ("attention", ["attn", "softmax", "flash", "decode_attn", "paged", "kv_cache"]),
    ("norm", ["norm", "rmsnorm", "layernorm", "rms_norm"]),
    ("gate-act", ["swiglu", "silu", "sigmoid", "gate", "act_"]),
    ("add-cast-misc", ["add", "cast", "copy", "memset", "transpose", "reshape",
                       "split", "concat", "embed", "sample", "topk", "rope"]),
]

TOK = 64.0  # graph 25 = one 64-tok rep; verify below
tot = defaultdict(float)
cnt = defaultdict(int)
other = defaultdict(float)
n_other = defaultdict(int)
for nid, s, e in con.execute(
        "SELECT demangledName, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE graphId=25"):
    nm = names.get(nid, "?")
    d = (e - s) / 1e6
    hit = False
    for lab, subs in PATS:
        nl = nm.lower()
        if any(x in nl for x in subs):
            tot[lab] += d
            cnt[lab] += 1
            hit = True
            break
    if not hit:
        other[nm[:80]] += d
        n_other[nm[:80]] += 1

print("== graph25 buckets, per-token (total/64) ==")
grand = 0.0
for lab, _ in PATS:
    pt = tot[lab] / TOK
    grand += pt
    print("%-14s %7.2f ms/tok  count=%d avg=%.1fus" % (
        lab, pt, cnt[lab], tot[lab] / max(cnt[lab], 1) / 1e3))
print("%-14s %7.2f" % ("SUM", grand))
print("== top unmatched kernels ==")
for nm, t in sorted(other.items(), key=lambda kv: -kv[1])[:15]:
    print("%7.2f %5d  %s" % (t / TOK, n_other[nm], nm))
