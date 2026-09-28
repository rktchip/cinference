import re
from collections import defaultdict

seqs = defaultdict(list)
for ln in open("/root/det.log", errors="replace"):
    m = re.search(r"\[serve-mtp\] seq=(\d+) lane=\d+ F=(\d+) .*?"
                  r"drafts=\[([^\]]*)\] verify=\[([^\]]*)\] accepted=(\d+)",
                  ln)
    if m:
        seqs[int(m.group(1))].append(
            (int(m.group(2)), m.group(3), m.group(4), int(m.group(5))))
print("seqs:", sorted(seqs))
for s in sorted(seqs):
    steps = seqs[s]
    print("seq=%d nsteps=%d total_accept=%d" % (
        s, len(steps), sum(a for _, _, _, a in steps)))
keys = sorted(seqs)
ref = keys[0]
for s in keys[1:]:
    a, b = seqs[ref], seqs[s]
    div = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
               min(len(a), len(b)))
    print("seq%d vs seq%d: first_div_step=%d/%d" % (s, ref, div,
                                                   min(len(a), len(b))))
    if div < min(len(a), len(b)):
        print("  ref:", a[div])
        print("  seq:", b[div])
