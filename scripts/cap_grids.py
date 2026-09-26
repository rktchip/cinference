import sqlite3

DB = "/root/speccoff_node.sqlite"
con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
names = dict(con.execute("SELECT id, value FROM StringIds"))

print("== grid geometry per linear kernel (graph25) ==")
q = ("SELECT demangledName, gridX, gridY, gridZ, blockX, blockY, blockZ, "
     "COUNT(*), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL "
     "WHERE graphId=25 GROUP BY demangledName, gridX, gridY, gridZ, blockX, "
     "blockY, blockZ ORDER BY 9 DESC LIMIT 40")
for nid, gx, gy, gz, bx, by, bz, n, t in con.execute(q):
    nm = names.get(nid, "?")
    nl = nm.lower()
    if any(x in nl for x in ["exl3", "gemm", "gemv", "cutlass", "had_", "hadamard",
                             "epilogue", "cast16", "bf16_to_fp16"]):
        print("%7.2f %5d grid=(%d,%d,%d) blk=(%d,%d,%d) %s" % (
            t / 1e6 / 64.0, n, gx, gy, gz, bx, by, bz, nm[:85]))
