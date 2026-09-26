import re, sys
from collections import Counter
logpath = sys.argv[1] if len(sys.argv) > 1 else "/root/s1A.log"
names = sys.argv[2:]
HIST = "--hist" in names
if HIST:
    names.remove("--hist")
log = open(logpath, errors="replace").read().splitlines()
steps = []
for line in log:
    m = re.search(r"\[mtp-step\] F=(\d+) a=(\d+) n=(\d+)", line)
    if m:
        steps.append(tuple(map(int, m.groups())))
trajs = []
for F, a, n in steps:
    if trajs and F < trajs[-1][-1][0] - 5:
        trajs.append([])
    if not trajs:
        trajs.append([])
    trajs[-1].append((F, a, n))
sel = trajs[-len(names):] if names else trajs
for i, t in enumerate(sel):
    name = names[i] if i < len(names) else ("traj%d" % i)
    acc = [a for _, a, _ in t]
    toks = sum(n for _, _, n in t)
    print("%s: steps=%d tokens=%d tok/step=%.2f mean-a=%.2f" %
          (name, len(t), toks, toks / len(t), sum(acc) / len(acc)))
    if HIST:
        print("   hist=%s" % dict(sorted(Counter(acc).items())))
