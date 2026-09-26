import sqlite3

DB = "/root/speccoff_node.sqlite"
con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)

print("== graphId histogram (count, total_ms) ==")
for g, n, t in con.execute(
        "SELECT graphId, COUNT(*), SUM(end-start) FROM "
        "CUPTI_ACTIVITY_KIND_KERNEL GROUP BY graphId ORDER BY 3 DESC"):
    print("graph=%s count=%d total_ms=%.2f" % (g, n, t / 1e6))

print("== top kernels overall (total_ms, count, avg_us) ==")
for nm, n, t in con.execute(
        "SELECT shortName, COUNT(*), SUM(end-start) FROM "
        "CUPTI_ACTIVITY_KIND_KERNEL GROUP BY shortName ORDER BY 3 DESC "
        "LIMIT 25"):
    print("%.2f %6d %8.1f  %s" % (t / 1e6, n, t / n / 1e3, nm))
