import re

lines = open("/root/split_nat.log", errors="replace").read().splitlines()
null_vals, id_vals = [], []
accept_diff = 0
logit_lines = 0
top1_mismatch = 0
last_legacy_top1 = None
for ln in lines:
    if "[slot-oracle]" not in ln:
        continue
    m = re.search(r"\bnull L=\d+ (\S+)", ln)
    if m:
        try:
            null_vals.append(float(m.group(1)))
        except ValueError:
            pass
    m = re.search(r"interdiff=\[([^\]]*)\]", ln)
    if m:
        for tok in m.group(1).split():
            try:
                v = float(tok.strip(","))
            except ValueError:
                continue
            if v >= 0:
                id_vals.append(v)
    if "ACCEPT-DIFF" in ln:
        accept_diff += 1
    m = re.search(r"(legacy|slots) logits top1=\[([^\]]*)\]", ln)
    if m:
        logit_lines += 1

print("null_n=%d null_max=%.5g null_nonzero=%d" % (
    len(null_vals), max(null_vals) if null_vals else -1,
    sum(1 for v in null_vals if v != 0)))
print("id_n=%d id_max=%.5g id_gt_0_25=%d id_gt_0_5=%d" % (
    len(id_vals), max(id_vals) if id_vals else -1,
    sum(1 for v in id_vals if v > 0.25),
    sum(1 for v in id_vals if v > 0.5)))
print("accept_diff=%d logit_lines=%d" % (accept_diff, logit_lines))
