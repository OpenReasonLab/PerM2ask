import argparse
import json
import sys
from pathlib import Path

from .bridge import verify


def main():
    p = argparse.ArgumentParser(
        description="Prepare principals for the dedicated HoneyBee benchmark database."
    )
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data"))
    args = p.parse_args()
    verify(args.export, args.data)
    up = args.upstream.resolve()
    sys.path.insert(0, str(up))
    from psycopg2 import sql
    from services.config import get_db_connection

    conf = json.loads((up / "config.json").read_text())
    if not conf.get("password"):
        raise ValueError("configure the password for this isolated benchmark service")
    principals = {
        json.loads(line)["user_id"]
        for line in (args.export / "queries.jsonl").read_text().splitlines()
    }
    conn = get_db_connection()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("SELECT user_id FROM Users")
            if {r[0] for r in cur.fetchall()} != principals:
                raise ValueError("dedicated DB principal set differs from export")
            cur.execute("SELECT rolname FROM pg_roles")
            if {str(u) for u in principals} & {r[0] for r in cur.fetchall()}:
                raise ValueError(
                    "existing numeric role names; use a new dedicated cluster"
                )
            cur.execute("SELECT comb_role, partition_id FROM CombRolePartitions")
            partitions = {}
            for comb, part in cur.fetchall():
                if len(comb) == 1:
                    partitions.setdefault(comb[0], set()).add(part)
            for uid in sorted(principals):
                role = sql.Identifier(str(uid))
                cur.execute(
                    sql.SQL("CREATE ROLE {} LOGIN PASSWORD %s").format(role),
                    (conf["password"],),
                )
                tables = [
                    "userroles",
                    "combrolepartitions",
                    *[
                        "documentblocks_partition_" + str(i)
                        for i in sorted(partitions.get(uid, ()))
                    ],
                ]
                cur.execute(
                    sql.SQL("GRANT SELECT ON {} TO {}").format(
                        sql.SQL(",").join(map(sql.Identifier, tables)), role
                    )
                )
    finally:
        conn.close()
    print(f"Prepared {len(principals)} logins for the dedicated benchmark database.")


if __name__ == "__main__":
    main()
