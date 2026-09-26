sqlite3 /root/speccoff_node.sqlite ".tables" 2>&1 | tr ' ' '\n' | grep -i "kernel\|memcpy\|runtime" | head -n 10
