import json, re, sys

streams = json.load(open("/root/det_detA.json"))
print("rep2==rep4:", streams[2] == streams[4])
print("BASE...:", repr(streams[0][60:115]))
print("REP2...:", repr(streams[2][60:115]))

gaps = []
for ln in open("/root/det.log", errors="replace"):
    m = re.search(r"\[mtp-gap\] F=(\d+) a=(\d+) tok=\d+ top=[\d.e+-]+ gap=(\S+)", ln)
    if m:
        gaps.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
print("gap_n=%d min=%.4g" % (len(gaps), min(g for _, _, g in gaps)))
tiny = sorted(g for _, _, g in gaps if g <= 0.5)
print("gaps<=0.5: n=%d vals=%s" % (len(tiny), [round(g, 4) for g in tiny][:20]))
print("zero_gap_Fs:", sorted(set(f for f, a, g in gaps if g == 0.0))[:20])
