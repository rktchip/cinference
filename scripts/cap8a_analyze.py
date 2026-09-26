#!/usr/bin/env python3
"""8a: slots-MTP step phase analysis from slotsmtp.sqlite. Usage: cap8a_analyze.py [stage]"""
import sqlite3, sys

DB = "file:/root/slotsmtp.sqlite?mode=ro"
c = sqlite3.connect(DB, uri=True)
stage = sys.argv[1] if len(sys.argv) > 1 else "bursts"

if stage == "bursts":
    st = [r[0] for r in c.execute(
        "SELECT n.start FROM NVTX_EVENTS n JOIN StringIds s ON n.textId=s.id "
        "WHERE (s.value LIKE 'verify.%' OR s.value LIKE 'mtp.%') ORDER BY n.start")]
    print("n ranges:", len(st), "span_s:", (st[-1] - st[0]) / 1e9)
    gaps = [((st[i + 1] - st[i]) / 1e9, st[i] / 1e9)
            for i in range(len(st) - 1) if (st[i + 1] - st[i]) > 2e8]
    print("gaps>0.2s (gap_s @at_s):")
    for g in gaps[-10:]:
        print("  %.2f @ %.1f" % g)

elif stage == "struct":
    t1 = float(sys.argv[2]); t2 = float(sys.argv[3])
    tids = c.execute(
        "SELECT n.globalTid, COUNT(*) FROM NVTX_EVENTS n JOIN StringIds s ON n.textId=s.id "
        "WHERE s.value='verify.layer.full' AND n.start>=? AND n.start<? "
        "GROUP BY 1", (int(t1 * 1e9), int(t2 * 1e9))).fetchall()
    print("verify.layer.full tids:", tids)
    tids2 = c.execute(
        "SELECT n.globalTid, COUNT(*) FROM NVTX_EVENTS n JOIN StringIds s ON n.textId=s.id "
        "WHERE s.value='mtp.forward' AND n.start>=? AND n.start<? "
        "GROUP BY 1", (int(t1 * 1e9), int(t2 * 1e9))).fetchall()
    print("mtp.forward tids:", tids2)
    print("first 8 verify.layer.full (start_ms, dur_us, tid):")
    for a, b, t in c.execute(
        "SELECT n.start, n.end, n.globalTid FROM NVTX_EVENTS n JOIN StringIds s ON n.textId=s.id "
        "WHERE s.value='verify.layer.full' AND n.start>=? AND n.start<? AND n.end IS NOT NULL "
        "ORDER BY n.start LIMIT 8", (int(t1 * 1e9), int(t2 * 1e9))).fetchall():
        print("  %.3f %d %s" % (a / 1e6, (b - a) / 1000, t))

elif stage == "pertid":
    t1 = float(sys.argv[2]); t2 = float(sys.argv[3])
    for tid in (281518043826731, 281518043826732):
        v = c.execute(
            "SELECT n.start, n.end FROM NVTX_EVENTS n JOIN StringIds s ON n.textId=s.id "
            "WHERE s.value='verify.layer.full' AND n.globalTid=? AND n.start>=? AND n.start<? "
            "AND n.end IS NOT NULL ORDER BY n.start",
            (tid, int(t1 * 1e9), int(t2 * 1e9))).fetchall()
        steps = []; cur = [v[0]]
        for a, b in zip(v, v[1:]):
            if b[0] - a[0] > 10e6:
                steps.append(cur); cur = [b]
            else:
                cur.append(b)
        steps.append(cur)
        print("tid %d: ranges=%d clusters=%d" % (tid, len(v), len(steps)))
        for s in steps:
            print("  n=%3d span=%7.2f ms" % (len(s), (s[-1][1] - s[0][0]) / 1e6))

elif stage == "gapdecomp":
    t1 = float(sys.argv[2]); t2 = float(sys.argv[3])
    tid = int(sys.argv[4])
    v = c.execute(
        "SELECT n.start, n.end FROM NVTX_EVENTS n JOIN StringIds s ON n.textId=s.id "
        "WHERE s.value='verify.layer.full' AND n.globalTid=? AND n.start>=? AND n.start<? "
        "AND n.end IS NOT NULL ORDER BY n.start",
        (tid, int(t1 * 1e9), int(t2 * 1e9))).fetchall()
    steps = []; cur = [v[0]]
    for a, b in zip(v, v[1:]):
        if b[0] - a[0] > 10e6:
            steps.append(cur); cur = [b]
        else:
            cur.append(b)
    steps.append(cur)
    steps = [s for s in steps if len(s) == 16]
    for i in range(1, min(len(steps) - 1, 8)):
        ga, gb = steps[i - 1][-1][1], steps[i][0][0]
        gk = c.execute(
            "SELECT SUM(end-start)/1e6, COUNT(*), COUNT(DISTINCT graphId) "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND start<? "
            "AND graphId IS NOT NULL", (ga, gb)).fetchone()
        ik = c.execute(
            "SELECT SUM(end-start)/1e6, COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "WHERE start>=? AND start<? AND graphId IS NULL", (ga, gb)).fetchone()
        print("gap%d wall=%.1f graph_kern=%.1fms n=%s launches=%s | indiv_kern=%.1fms n=%s" % (
            i, (gb - ga) / 1e6, gk[0] or 0, gk[1], gk[2], ik[0] or 0, ik[1]))
    print("--- d2h blocking (bytes, gap_to_next_kernel_us):")
    for i in range(1, 4):
        ga, ve = steps[i - 1][-1][1], steps[i][-1][1]
        for (ms, me, by) in c.execute(
            "SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY "
            "WHERE start>=? AND start<? AND copyKind=2 ORDER BY start", (ga, ve)).fetchall():
            nk = c.execute(
                "SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=?",
                (me,)).fetchone()[0]
            print("  step%d d2h bytes=%6d followed_gap_us=%d" % (i, by, ((nk or me) - me) // 1000))

elif stage == "phase":
    t1 = float(sys.argv[2]); t2 = float(sys.argv[3])
    tid = int(sys.argv[4])
    v = c.execute(
        "SELECT n.start, n.end FROM NVTX_EVENTS n JOIN StringIds s ON n.textId=s.id "
        "WHERE s.value='verify.layer.full' AND n.globalTid=? AND n.start>=? AND n.start<? "
        "AND n.end IS NOT NULL ORDER BY n.start",
        (tid, int(t1 * 1e9), int(t2 * 1e9))).fetchall()
    steps = []; cur = [v[0]]
    for a, b in zip(v, v[1:]):
        if b[0] - a[0] > 10e6:
            steps.append(cur); cur = [b]
        else:
            cur.append(b)
    steps.append(cur)
    steps = [s for s in steps if len(s) == 16]
    print("tid %d: 16-range steps=%d" % (tid, len(steps)))
    for i in range(1, min(len(steps) - 1, 8)):
        vs, ve = steps[i][0][0], steps[i][-1][1]
        ga, gb = steps[i - 1][-1][1], steps[i][0][0]
        vspan = (ve - vs) / 1e6
        k = c.execute(
            "SELECT COUNT(*), SUM(end-start)/1e6, SUM(graphId IS NOT NULL) "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND start<?", (vs, ve)).fetchone()
        g = c.execute(
            "SELECT COUNT(*), SUM(n.end-n.start)/1e6 FROM NVTX_EVENTS n "
            "JOIN StringIds s ON n.textId=s.id WHERE s.value='verify.layer.gdn' "
            "AND n.start>=? AND n.start<? AND n.end IS NOT NULL", (vs, ve)).fetchone()
        k2 = c.execute(
            "SELECT COUNT(*), SUM(end-start)/1e6, SUM(graphId IS NOT NULL) "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND start<?", (ga, gb)).fetchone()
        f = c.execute(
            "SELECT COUNT(*), SUM(n.end-n.start)/1e6 FROM NVTX_EVENTS n "
            "JOIN StringIds s ON n.textId=s.id WHERE s.value='mtp.forward' "
            "AND n.start>=? AND n.start<? AND n.end IS NOT NULL", (ga, gb)).fetchone()
        g2 = c.execute(
            "SELECT COUNT(*) FROM NVTX_EVENTS n "
            "JOIN StringIds s ON n.textId=s.id WHERE s.value='verify.layer.gdn' "
            "AND n.start>=? AND n.start<?", (ga, gb)).fetchone()
        m = c.execute(
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
            "WHERE start>=? AND start<? AND copyKind=2", (ga, ve)).fetchone()
        print("step%d verify_span=%.1f kverify=%s/%s graph=%s gdn_in=%s/%.1f | gap=%.1f kgap=%s/%s g=%s fwd=%s/%.1f gdn_gap=%s d2h=%s" % (
            i, vspan, k[0], round(k[1] or 0, 1), k[2], g[0], round(g[1] or 0, 1),
            (gb - ga) / 1e6, k2[0], round(k2[1] or 0, 1), k2[2], f[0], round(f[1] or 0, 1),
            g2[0], m[0]))

elif stage == "names":
    t1 = float(sys.argv[2]); t2 = float(sys.argv[3])
    q = c.execute(
        "SELECT s.value, COUNT(*), SUM(n.end-n.start)/1e6 FROM NVTX_EVENTS n "
        "JOIN StringIds s ON n.textId=s.id "
        "WHERE n.start>=? AND n.start<? AND n.end IS NOT NULL "
        "GROUP BY s.value ORDER BY 3 DESC", (int(t1 * 1e9), int(t2 * 1e9)))
    print("ranges in [%.1f, %.1f]:" % (t1, t2))
    for name, n, ms in q.fetchall():
        print("  %-28s %6d  %9.2f ms total" % (name, n, ms))
