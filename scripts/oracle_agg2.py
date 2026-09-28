import re

lines = open("/root/split_nat.log", errors="replace").read().splitlines()
nok = sum(1 for ln in lines if "NULL-OK" in ln)
nvoid = sum(1 for ln in lines if "NULL-DIFF-VOID" in ln)
rok = sum(1 for ln in lines if "RESTORE-OK" in ln)
rdiff = sum(1 for ln in lines if "RESTORE-DIFF" in ln)
print("null_ok=%d null_void=%d restore_ok=%d restore_diff=%d" % (
    nok, nvoid, rok, rdiff))
ad = [ln for ln in lines if "ACCEPT-DIFF" in ln]
print("accept_diff_n=%d" % len(ad))
for ln in ad:
    m = re.search(r"F=(\d+) lane=(\d+) a_leg=(\d+) a_slot=(\d+)", ln)
    print("F=%s lane=%s a_leg=%s a_slot=%s" % m.groups())
gaps = {}
for ln in lines:
    m = re.search(r"\[mtp-gap\] F=(\d+) a=(\d+) tok=\d+ top=[\d.e+-]+ gap=(\S+)", ln)
    if m:
        gaps.setdefault(int(m.group(1)), []).append(
            (int(m.group(2)), float(m.group(3))))
for ln in ad:
    m = re.search(r"F=(\d+)", ln)
    f = int(m.group(1))
    print("F=%d gaps=%s" % (f, gaps.get(f, "NONE")))
mx = 0.0
per_layer = {}
for ln in lines:
    m = re.search(r"\[slot-oracle\] null L=(\d+) (\S+)", ln)
    if m:
        try:
            v = float(m.group(1 + 1))
        except ValueError:
            continue
        per_layer.setdefault(int(m.group(1)), []).append(v)
        mx = max(mx, v)
print("null_max=%.4g" % mx)
worst = sorted(((max(v), l, len(v)) for l, v in per_layer.items()),
               reverse=True)[:6]
print("worst_layers(max,n):", [(l, round(m, 4), n) for m, l, n in worst])
