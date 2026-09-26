import sqlite3

DB = "/root/speccoff_node.sqlite"
con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)

print("== distinct graphIds ==")
print(con.execute(
    "SELECT graphId, COUNT(*), MIN(start), MAX(end) FROM "
    "CUPTI_ACTIVITY_KIND_KERNEL GROUP BY graphId").fetchall())
print("== runtime table present? ==")
print(con.execute(
    "SELECT name FROM sqlite_master WHERE type='table'").fetchall())
