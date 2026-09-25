#!/usr/bin/env python3
"""Off-GPU permutation test for the row-20b conv-window layout bug.

Reads /root/oracle_win_F<a>.bin (u32 nlayers; per layer: u32 L, u64 m,
m floats T-chain, m floats A-canonical; layout linear t*C+ch, T=3 slices).
Per layer:
  - diff stats (max/mean, NaN-aware) for identity, time-flip, transpose,
    transpose-flip, and the two one-slot overlaps;
  - 3x3 time-slot Pearson matrix (each t-slot vs each canonical slot,
    across channels): diagonal = identity, anti-diagonal = flip,
    off-diagonal = shift;
  - 3x3 segment matrix (q/k/v blocks) when the time matrix is weak.
A layout match lands within family noise (~1e-2); a wrong layout sits at
activation scale (~1). A true no-match after both tools means the layout
class itself is wrong (ticket: reopens the investigation).

Pure stdlib (no numpy anywhere near this box). Usage:
  python3 winperm.py /root/oracle_win_F53_a1.bin
"""
import math
import struct
import sys

# EXL3 GDN dims (wrapper): q/k/v channel segments.
SEGS = (("q", 0, 2048), ("k", 2048, 4096), ("v", 4096, 10240))


def read_layer(f):
    head = f.read(12)
    if len(head) < 12:
        return None
    (lid, m) = struct.unpack("<IQ", head)
    want = 4 * m
    tb = f.read(want)
    ab = f.read(want)
    if len(tb) < want or len(ab) < want:
        return None
    t = struct.unpack("<%df" % m, tb)
    a = struct.unpack("<%df" % m, ab)
    return (lid, m, t, a)


def is_nan(x):
    return x != x


def diff_stats(pairs):
    n = 0
    worst = 0.0
    s = 0.0
    nn = 0
    for (x, y) in pairs:
        if is_nan(x) or is_nan(y):
            nn += 1
            continue
        d = abs(x - y)
        s += d
        n += 1
        if d > worst:
            worst = d
    return (worst, s / n if n else float("nan"), nn)


def pearson(xs, ys):
    n = 0
    sx = sy = sxx = syy = sxy = 0.0
    for (x, y) in zip(xs, ys):
        if is_nan(x) or is_nan(y):
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


def slot(t, c, tt, ch):
    return t[tt * c + ch]


def analyze(lid, m, t, a):
    c = m // 3
    out = {}
    out["id"] = diff_stats(zip(t, a))
    out["flip"] = diff_stats((slot(t, c, 2 - tt, ch), slot(a, c, tt, ch))
                             for tt in range(3) for ch in range(c))
    # transpose: chain wrote A^T -> T[t,c] holds A[c,t]
    out["tr"] = diff_stats((t[tt * c + ch], a[ch * 3 + tt])
                           for tt in range(3) for ch in range(c))
    out["trflip"] = diff_stats((t[tt * c + ch], a[ch * 3 + (2 - tt)])
                               for tt in range(3) for ch in range(c))
    out["sh_lo"] = diff_stats((t[k], a[c + k]) for k in range(2 * c))
    out["sh_hi"] = diff_stats((t[c + k], a[k]) for k in range(2 * c))
    # 3x3 time mapping matrix.
    tm = [[pearson((slot(t, c, ti, ch) for ch in range(c)),
                   (slot(a, c, aj, ch) for ch in range(c)))
           for aj in range(3)] for ti in range(3)]
    out["tm"] = tm
    tmax = max((v for row in tm for v in row if v == v), default=float("nan"))
    out["tmax"] = tmax
    # segment matrix: always printed (a partial reorder, e.g. k/v swapped
    # with q in place, leaves a moderately strong time diagonal that a
    # "weak" gate would hide behind).
    sm = [[pearson((t[tt * c + ch] for tt in range(3)
                    for ch in range(s0, s1)),
                   (a[tt * c + ch] for tt in range(3)
                    for ch in range(d0, d1)))
           for (dn, d0, d1) in SEGS] for (sn, s0, s1) in SEGS]
    out["sm"] = sm
    return out


def fmt_r(v):
    return "  -- " if v != v else "%+.2f" % v


def main(path):
    print("per-layer: diff max/mean (id flip tr trflip shi_lo shi_hi) + time 3x3 (rows=t,cols=a)")
    cands = ("id", "flip", "tr", "trflip", "sh_lo", "sh_hi")
    with open(path, "rb") as f:
        (nl,) = struct.unpack("<I", f.read(4))
        for _ in range(nl):
            r = read_layer(f)
            if r is None:
                print("SHORT READ")
                break
            (lid, m, t, a) = r
            r_ = analyze(lid, m, t, a)
            cells = " ".join("%.3g/%.3g" % (r_[k][0], r_[k][1]) for k in cands)
            print("L%3d C%-6d %s" % (lid, m // 3, cells))
            print("   time " + " ".join(fmt_r(v) for row in r_["tm"] for v in row) +
                  "  (tmax=%.2f)" % (r_["tmax"] if r_["tmax"] == r_["tmax"] else -2.0))
            if r_["sm"] is not None:
                names = [s[0] for s in SEGS]
                for (i, row) in enumerate(r_["sm"]):
                    print("   seg%s " % names[i] +
                          " ".join(fmt_r(v) for v in row))
    print("DONE layers=%d (match = ~1e-2 scale; wrong layout = ~1 scale)" % nl)


if __name__ == "__main__":
    main(sys.argv[1])
