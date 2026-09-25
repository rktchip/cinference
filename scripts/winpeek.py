#!/usr/bin/env python3
"""Print raw values around the X anomaly (T[2] slice) for eyeball triage."""
import struct
import sys


def main(path, layer, n):
    with open(path, "rb") as f:
        (nl,) = struct.unpack("<I", f.read(4))
        if layer < 0:
            # stat mode: per-layer max/mean |T2-A2| + self-similarity
            # (exact-frac A0/A1, A1/A2, T0/T1) to close the A1==A2 loophole.
            print("per-layer |X-A2|: max / mean + self exact-fracs")
            for _ in range(nl):
                (lid, m) = struct.unpack("<IQ", f.read(12))
                t = struct.unpack("<%df" % m, f.read(4 * m))
                a = struct.unpack("<%df" % m, f.read(4 * m))
                c = m // 3
                worst = 0.0
                s = 0.0
                nn = 0
                for i in range(c):
                    x = t[2 * c + i]
                    y = a[2 * c + i]
                    if x != x or y != y:
                        nn += 1
                        continue
                    d = abs(x - y)
                    s += d
                    if d > worst:
                        worst = d

                def ef(u, v):
                    n = tot = 0
                    for (x, y) in zip(u, v):
                        if x != x or y != y:
                            continue
                        tot += 1
                        if x == y:
                            n += 1
                    return n / tot if tot else float("nan")

                print("  L%3d max=%.4g mean=%.4g nan=%d self[A01=%.3f A12=%.3f T01=%.3f]" %
                      (lid, worst, s / c, nn, ef(a[0:c], a[c:2 * c]),
                       ef(a[c:2 * c], a[2 * c:3 * c]), ef(t[0:c], t[c:2 * c])))
            return
        for _ in range(nl):
            (lid, m) = struct.unpack("<IQ", f.read(12))
            t = struct.unpack("<%df" % m, f.read(4 * m))
            a = struct.unpack("<%df" % m, f.read(4 * m))
            if lid != layer:
                continue
            c = m // 3
            x = t[2 * c:3 * c]
            a2 = a[2 * c:3 * c]
            print("L%d C=%d  first %d of X(T2) vs A2:" % (lid, c, n))
            for i in range(n):
                print("  [%4d] X=%+.6f A2=%+.6f d=%+.6f" % (i, x[i], a2[i], x[i] - a2[i]))
            # zero/const pattern in X?
            xn = [v for v in x if v == v]
            print("X: zeros=%d mean=%+.4g maxabs=%.4g" %
                  (sum(1 for v in xn if v == 0.0), sum(xn) / len(xn),
                   max(abs(v) for v in xn)))
            return
    print("layer not found")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 30)
