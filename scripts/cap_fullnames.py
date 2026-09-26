import sqlite3

DB = "/root/speccoff_node.sqlite"
con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
names = dict(con.execute("SELECT id, value FROM StringIds"))

seen = {}
for nid, gx, gy, gz, n, t in con.execute(
        "SELECT demangledName, gridX, gridY, gridZ, COUNT(*), SUM(end-start)"
        " FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId=25"
        " GROUP BY demangledName, gridX, gridY, gridZ"):
    nm = names.get(nid, "?")
    nl = nm.lower()
    if any(x in nl for x in ["exl3", "gemm", "gemv", "cutlass", "had_",
                             "hadamard", "epilogue", "cast16",
                             "bf16_to_fp16"]):
        seen.setdefault(nm, []).append((gx, gy, gz, n, t))

out = []
for nm, occ in sorted(seen.items()):
    out.append("=" * 100)
    out.append(nm[:600])
    for gx, gy, gz, n, t in occ:
        out.append("    grid=(%d,%d,%d) n=%d us/tok=%.2f" % (
            gx, gy, gz, n, t / 1000.0 / 64.0))
open("/tmp/fullnames.txt", "w").write("\n".join(out) + "\n")
print("WROTE %d kernels" % len(seen))
