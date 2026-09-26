import sqlite3

DB = "/root/speccoff_node.sqlite"
con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)

print("== StringIds schema ==")
print(con.execute(
    "SELECT sql FROM sqlite_master WHERE name='StringIds'").fetchone()[0])
print("== sample ==")
for r in con.execute("SELECT * FROM StringIds LIMIT 3"):
    print(r)
