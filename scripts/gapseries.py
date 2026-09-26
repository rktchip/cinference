import re, sys
log = open("/root/mtpA.log", errors="replace").read()
cur = None
rows = []
pat1 = re.compile(r"slots logits top1=\[([^\]]*)\] gap=\[([^\]]*)\]")
pat2 = re.compile(r"\[slot-oracle\] F=(\d+) lane=(\d+) a_leg=(\d+) a_slot=(\d+) (\S+)")
for line in log.splitlines():
    m = pat1.search(line)
    if m:
        cur = m.group(2)
    m2 = pat2.search(line)
    if m2 and cur:
        g0 = cur.split()[0]
        rows.append((int(m2.group(1)), int(m2.group(2)), m2.group(3),
                     m2.group(4), m2.group(5), g0))
        cur = None
for r in rows:
    print("F=%d lane=%d leg=%s slot=%s %s gap0=%s" % r)
