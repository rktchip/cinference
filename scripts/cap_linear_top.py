import sqlite3

DB = "/root/speccoff_node.sqlite"
con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
names = dict(con.execute("SELECT id, value FROM StringIds"))

print("== linear-bucket top names, graph25, per-64tok ==")
rows = []
for nid, n, t in con.execute(
        "SELECT demangledName, COUNT(*), SUM(end-start) FROM "
        "CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId=25 "
        "GROUP BY demangledName ORDER BY 3 DESC LIMIT 60"):
    nm = names.get(nid, "?")
    nl = nm.lower()
    if any(x in nl for x in ["exl3", "quant_", "gemm", "gemv", "cutlass",
                             "nvfp4", "trellis", "dequant"]):
        rows.append((t / 1e6 / 64.0, n, t / n / 1e3, nm[:110]))
for r in rows[:18]:
    print("%6.2f %6d %7.1f  %s" % r)
print("linear-name-count:", len(rows))
print("== kernels/token in graph25:", 120897 / 64.0)
print("== distinct streamIds in graph25:",
      con.execute("SELECT COUNT(DISTINCT streamId) FROM "
                  "CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId=25").fetchone()[0])
