import struct, sys
def load(p):
    out = {}
    with open(p, "rb") as f:
        (nl,) = struct.unpack("<I", f.read(4))
        for _ in range(nl):
            (lid, m) = struct.unpack("<IQ", f.read(12))
            out[lid] = struct.unpack("<%df" % m, f.read(4 * m))
    return out
B = "/root/oracle_CTX_F53_a1"
pre = load(B + "_pre.bin"); mid = load(B + "_mid.bin")
t2 = load(B + "_t2.bin"); sh2 = load(B + "_sh2.bin")
m = len(pre[0]) // 3
for name, buf in (("pre", pre), ("mid", mid), ("t2", t2), ("sh2", sh2)):
    L0 = buf[0]
    print(name, "m=", m,
          "s0[:4]=", ["%.4g" % v for v in L0[0:4]],
          "s1[:4]=", ["%.4g" % v for v in L0[m:m + 4]],
          "s2[:4]=", ["%.4g" % v for v in L0[2 * m:2 * m + 4]])
# Note 1 three-way: pre s0-vs-s2 per layer (exact-frac + max).
# coherent-diff-positions -> 1.0 at L0, decaying with depth;
# stale-same-token -> flat small; aliasing (same bytes) -> 1.0 everywhere.
print("pre s0-vs-s2 by layer:")
for lid in sorted(pre):
    a = pre[lid]
    n = len(a) // 3
    e = sum(1 for i in range(n) if a[i] == a[2 * n + i]) / n
    mx = max(abs(a[i] - a[2 * n + i]) for i in range(n))
    print(" L=%u ef=%.4f max=%.4g" % (lid, e, mx))
