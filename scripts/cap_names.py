import sqlite3

DB = "/root/speccoff_node.sqlite"
con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)

print("== graph 25 (per-64-tok: total_ms/64) top 20 ==")
for nm, n, t in con.execute(
        "SELECT demangledName, COUNT(*), SUM(end-start) FROM "
        "CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId=25 "
        "GROUP BY demangledName ORDER BY 3 DESC LIMIT 20"):
    print("%7.2f %6d %7.1f  %s" % (t / 1e6, n, t / n / 1e3, (nm or "?")[:100]))
