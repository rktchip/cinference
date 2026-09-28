import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/root/split_nat.log"
lines = open(path, errors="replace").read().splitlines()
nok = sum(1 for ln in lines if "NULL-OK" in ln)
nvoid = sum(1 for ln in lines if "NULL-DIFF-VOID" in ln)
rok = sum(1 for ln in lines if "RESTORE-OK" in ln)
rdiff = sum(1 for ln in lines if "RESTORE-DIFF" in ln)
ad = [ln for ln in lines if "ACCEPT-DIFF" in ln]
print("%s null_ok=%d null_void=%d restore_ok=%d restore_diff=%d "
      "accept_diff=%d" % (path, nok, nvoid, rok, rdiff, len(ad)))
for ln in ad[:8]:
    m = re.search(r"F=(\d+) lane=(\d+) a_leg=(\d+) a_slot=(\d+)", ln)
    print("  F=%s lane=%s a_leg=%s a_slot=%s" % (
        m.groups() if m else ("?", "?", "?", "?")))
id_vals = []
for ln in lines:
    m = re.search(r"interdiff=\[([^\]]*)\]", ln)
    if m:
        for tok in m.group(1).split():
            try:
                v = float(tok.strip(","))
            except ValueError:
                continue
            if v >= 0:
                id_vals.append(v)
print("id_n=%d id_max=%.4g id_gt_0_25=%d" % (
    len(id_vals), max(id_vals) if id_vals else -1,
    sum(1 for v in id_vals if v > 0.25)))
