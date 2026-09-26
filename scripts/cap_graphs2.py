import sqlite3

DB = "/root/speccoff_node.sqlite"
con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
rows = con.execute(
    "SELECT graphId, COUNT(*), MIN(start), MAX(end) FROM "
    "CUPTI_ACTIVITY_KIND_KERNEL GROUP BY graphId").fetchall()
t0 = min(r[2] for r in rows)
out = ["ngraphs=%d" % len(rows)]
for gid, n, mn, mx in rows:
    out.append("graph=%s n=%d t0+%dms..+%dms span=%.1fms" % (
        gid, n, (mn - t0) // 1000000, (mx - t0) // 1000000,
        (mx - mn) / 1e6))
open("/tmp/graphs.txt", "w").write("\n".join(out) + "\n")
print("WROTE", len(out))
