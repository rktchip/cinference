#!/usr/bin/env python3
"""Deep hunt on one oracle window dump: exact-match slice matrix per layer,
X-slice (T[2]) characterization, and subsampled cross-layer correlation.

Usage: python3 winhunt.py /root/oracle_win_F53_a1.bin [maxlayers]
Pure stdlib.
"""
import math
import struct
import sys


def load(path):
    layers = []
    with open(path, "rb") as f:
        (nl,) = struct.unpack("<I", f.read(4))
        for _ in range(nl):
            (lid, m) = struct.unpack("<IQ", f.read(12))
            t = struct.unpack("<%df" % m, f.read(4 * m))
            a = struct.unpack("<%df" % m, f.read(4 * m))
            layers.append((lid, m, t, a))
    return layers


def sl(buf, c, tt):
    return buf[tt * c:(tt + 1) * c]


def exact_frac(u, v):
    n = 0
    tot = 0
    for (x, y) in zip(u, v):
        if x != x or y != y:
            continue
        tot += 1
        if x == y:
            n += 1
    return n / tot if tot else float("nan")


def pearson_sub(u, v, stride):
    n = 0
    sx = sy = sxx = syy = sxy = 0.0
    for k in range(0, len(u), stride):
        x = u[k]
        y = v[k]
        if x != x or y != y:
            continue
        n += 1
        sx += x
        sy += y
        sxx += x * x
        syy += y * y
        sxy += x * y
    if n < 3:
        return float("nan")
    cov = sxy - sx * sy / n
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def main(path, maxlayers):
    layers = load(path)
    print("layers=%d" % len(layers))
    show = layers[:maxlayers]
    for (lid, m, t, a) in show:
        c = m // 3
        mat = [[exact_frac(sl(t, c, ti), sl(a, c, aj)) for aj in range(3)]
               for ti in range(3)]
        print("L%3d exact-frac rows=t cols=a:" % lid +
              " ".join("%.3f" % v for row in mat for v in row))
    # X characterization on layer 0: distinct count, mean, and exact hit vs A2.
    (lid, m, t, a) = layers[0]
    c = m // 3
    x = [v for v in sl(t, c, 2) if v == v]
    y = [v for v in sl(a, c, 2) if v == v]
    print("L0 X: n=%d distinct=%d mean=%.4g | A2: mean=%.4g | X==A2 exact=%.4f" %
          (len(x), len(set(x)), sum(x) / len(x), sum(y) / len(y),
           exact_frac(sl(t, c, 2), sl(a, c, 2))))
    # cross-layer: X(L0) vs A-mid slices of every layer (stride 16).
    print("L0-X vs every-layer A slices r (stride16): [a0,a1,a2] top5 by max|r|")
    res = []
    for (lid2, m2, t2, a2) in layers:
        c2 = m2 // 3
        r = [pearson_sub(sl(t, c, 2)[::1], sl(a2, c2, aj)[::1], 16) for aj in range(3)]
        res.append((lid2, r))
    res.sort(key=lambda kv: -max(abs(v) for v in kv[1] if v == v))
    for (lid2, r) in res[:5]:
        print("  L%3d %+.3f %+.3f %+.3f" % (lid2, r[0], r[1], r[2]))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)
