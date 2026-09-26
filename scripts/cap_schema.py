import sqlite3, re

DB = "/root/speccoff_node.sqlite"
con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
tabs = [r[0] for r in con.execute(
    "SELECT name FROM sqlite_master WHERE type='table'")]
print("TABLES:", " ".join(tabs))
kt = None
for t in tabs:
    if "KERNEL" in t.upper():
        kt = t
        break
print("KERNEL_TABLE:", kt)
cols = [r[1] for r in con.execute("PRAGMA table_info(%s)" % kt)]
print("COLS:", " ".join(cols))
