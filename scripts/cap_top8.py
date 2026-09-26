import sqlite3

DB = "/root/speccoff_node.sqlite"
con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
names = dict(con.execute("SELECT id, value FROM StringIds"))

rows = []
for nid, n, t in con.execute(
        "SELECT demangledName, COUNT(*), SUM(end-start) FROM "
        "CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId=25 "
        "GROUP BY demangledName ORDER BY 3 DESC LIMIT 8"):
    rows.append((t / 1e6 / 64.0, n, t / n / 1e3, names.get(nid, "?")[:110]))
for ms, n, avg, nm in rows:
    print("%6.2f %6d %7.1f  %s" % (ms, n, avg, nm))
