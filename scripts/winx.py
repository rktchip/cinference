#!/usr/bin/env python3
"""Compare oracle CTX singles: slice exact-frac + diff stats between named
conv buffers. Usage:
  python3 winx.py /root/oracle_CTX_F53_a1 pre mid
  python3 winx.py /root/oracle_CTX_F53_a1 t2 full
Layout per buffer: linear t*C+ch, T=3 slices. Pure stdlib.
"""
import struct
import sys


def load_single(path):
    layers = {}
    with open(path, "rb") as f:
        (nl,) = struct.unpack("<I", f.read(4))
        for _ in range(nl):
            (lid, m) = struct.unpack("<IQ", f.read(12))
            buf = struct.unpack("<%df" % m, f.read(4 * m))
            layers[lid] = buf
    return layers


def sl(buf, c, tt):
    return buf[tt * c:(tt + 1) * c]


def ef(u, v):
    n = tot = 0
    for (x, y) in zip(u, v):
        if x != x or y != y:
            continue
        tot += 1
        if x == y:
            n += 1
    return n / tot if tot else float("nan")


def dm(u, v):
    worst = 0.0
    s = 0.0
    n = 0
    for (x, y) in zip(u, v):
        if x != x or y != y:
            continue
        d = abs(x - y)
        s += d
        n += 1
        if d > worst:
            worst = d
    return (worst, s / n if n else float("nan"))


def main(prefix, ta, tb):
    a = load_single(prefix + "_" + ta + ".bin")
    b = load_single(prefix + "_" + tb + ".bin")
    print("%s vs %s: per-layer slice exact-frac (rows=%s cols=%s) + full max/mean" %
          (ta, tb, ta, tb))
    for lid in sorted(a):
        if lid not in b:
            continue
        u = a[lid]
        v = b[lid]
        c = len(u) // 3
        mat = [[ef(sl(u, c, ti), sl(v, c, aj)) for aj in range(3)] for ti in range(3)]
        (w, m) = dm(u, v)
        print("L%3d " % lid + " ".join("%.3f" % x for row in mat for x in row) +
              " | full max=%.4g mean=%.4g" % (w, m))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
