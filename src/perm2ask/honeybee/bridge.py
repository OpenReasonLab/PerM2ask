import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

from ..benchmark import Benchmark
from ..reference import read_audit
from ..reference import sha256 as digest

PIN = "efc1911aa85d1b02593af9c117af7caa01e5c4ed"


def records(path):
    with path.open(encoding="utf-8-sig") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def dump_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def csv_rows(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as fp:
        wr = csv.writer(fp)
        wr.writerow(fields)
        wr.writerows(rows)


def vector(encoder, text):
    arr = encoder.encode([text])[0]
    if len(arr) != 384 or not all(math.isfinite(float(x)) for x in arr):
        raise ValueError("invalid BGE vector")
    return "[" + ",".join(format(float(x), ".9g") for x in arr) + "]"


def sources(data):
    return {
        "requests": data / "benchmark/requests.jsonl",
        "subjects": data / "benchmark/subjects.jsonl",
        "policy": data / "policy.json",
        "notifications": data / "trusted/notifications.jsonl",
        "grants": data / "trusted/grants.jsonl",
        "audit": data / "audit.json",
    }


def export(data, encoder_path, dest, limit):
    from ..method import FrozenBGE

    source = sources(data)
    source_hashes = {k: digest(v) for k, v in source.items()}
    audit, counts = read_audit(source["audit"])
    for key in ("requests", "subjects"):
        if source_hashes[key] != audit["sha256"][f"benchmark/{key}.jsonl"]:
            raise ValueError(f"{key} snapshot differs from versioned audit")
    if source_hashes["policy"] != audit["sha256"]["policy.json"]:
        raise ValueError("policy snapshot differs from versioned audit")
    for key in ("notifications", "grants"):
        if source_hashes[key] != audit["sha256"][f"trusted/{key}.jsonl"]:
            raise ValueError(f"{key} snapshot differs from versioned audit")
    db = Benchmark(data)
    requests = [r for r in records(source["requests"]) if r["split"] == "test"]
    if Counter(r["dataset"] for r in requests) != {
        k: sum(v.values()) for k, v in counts.items()
    }:
        raise ValueError(
            "workload sizes differ from versioned per-dataset test denominators"
        )
    if limit:
        requests = requests[:limit]
    if not requests or len(set(r["request_id"] for r in requests)) != len(requests):
        raise ValueError("empty workload or duplicate request IDs")
    bindings = {}
    proofs = {}
    for r in requests:
        case = db.case_for(r)
        rid = r["request_id"]
        bindings[rid] = (
            r["subject_id"],
            r["scenario_id"],
            tuple(sorted(case["authorized_ids"])),
        )
        if "authorization_digest" in case:
            proofs[rid] = case["authorization_digest"]
    actors = sorted(set(bindings.values()))
    principals = {actor: i for i, actor in enumerate(actors, 1)}
    encoder = FrozenBGE(encoder_path)
    facts = []
    blocks = []
    principal_permissions = []
    actors_by_subject = {}
    for (subject, _, scope), uid in principals.items():
        actors_by_subject.setdefault(subject, []).append((set(scope), uid))
    for subject_id in sorted(actors_by_subject):
        s = db.subjects[subject_id]
        docs = {d["document_id"]: d["text"] for d in s["documents"]}
        for f in sorted(s["facts"], key=lambda f: f["fact_id"]):
            if docs[f["document_id"]][f["start"] : f["end"]] != f["text"]:
                raise ValueError("invalid source span")
            i = len(facts) + 1
            facts.append(
                dict(
                    db_document_id=i,
                    db_block_id=i,
                    subject_id=subject_id,
                    fact_id=f["fact_id"],
                    field_id=f["field_id"],
                    doc_id=f["document_id"],
                    text=f["text"],
                    forbidden_joint_sets=s.get("forbidden_joint_sets", []),
                )
            )
            raw = f["text"].encode("utf-8")
            blocks.append(
                (
                    i,
                    i,
                    "\\x" + raw.hex(),
                    "\\x" + hashlib.sha256(raw).hexdigest(),
                    vector(encoder, f["text"]),
                )
            )
            for scope, uid in actors_by_subject[subject_id]:
                if f["fact_id"] in scope:
                    principal_permissions.append((uid, i))
    queries = []
    for r in requests:
        rid = r["request_id"]
        uid = principals[bindings[rid]]
        queries.append(
            dict(
                request_id=r["request_id"],
                subject_id=r["subject_id"],
                scenario_id=r["scenario_id"],
                user_id=uid,
                query_text=r["question"],
                query_vector=vector(encoder, r["question"]),
                **({"authorization_digest": proofs[rid]} if rid in proofs else {}),
            )
        )
    if dest.exists():
        raise FileExistsError(f"use a new export path: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as temporary:
        stage = Path(temporary) / "data"
        stage.mkdir()
        dump_rows(stage / "facts.jsonl", facts)
        dump_rows(stage / "queries.jsonl", queries)
        csv_rows(
            stage / "Users.csv",
            ["user_id", "user_name"],
            [(i, f"{a[0]}/{a[1]}/{i}") for a, i in principals.items()],
        )
        csv_rows(
            stage / "Roles.csv",
            ["role_id", "role_name"],
            [(i, f"{a[0]}/{a[1]}/{i}") for a, i in principals.items()],
        )
        csv_rows(
            stage / "UserRoles.csv",
            ["user_id", "role_id"],
            [(i, i) for i in principals.values()],
        )
        csv_rows(
            stage / "Documents.csv",
            ["document_id", "document_name"],
            [(r["db_document_id"], r["fact_id"]) for r in facts],
        )
        csv_rows(
            stage / "DocumentBlocks.csv",
            ["block_id", "document_id", "block_content", "hash_value", "vector"],
            blocks,
        )
        csv_rows(
            stage / "PermissionAssignment.csv",
            ["role_id", "document_id"],
            principal_permissions,
        )
        if source_hashes != {k: digest(v) for k, v in source.items()}:
            raise ValueError(
                "dataset changed during export; retry using the frozen snapshot"
            )
        manifest = dict(
            upstream_commit=PIN,
            dataset_version=audit["version"],
            reference_denominators=counts,
            scope="test" if not limit else "test_subset",
            requests=len(queries),
            fact_documents=len(facts),
            principals=len(principals),
            vector_dimension=384,
            encoder_signature=encoder.signature,
            sources=source_hashes,
            files={p.name: digest(p) for p in sorted(stage.iterdir())},
        )
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        stage.rename(dest)
    print(f"Exported {len(queries)} queries and {len(facts)} fact blocks to {dest}")


def verify(folder, data):
    man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    audit, counts = read_audit(data / "audit.json")
    if (
        man.get("dataset_version") != audit["version"]
        or man.get("reference_denominators") != counts
    ):
        raise ValueError("export is not bound to this expanded benchmark")
    if man["scope"] == "test" and man["requests"] != audit["request_count"]:
        raise ValueError("incomplete full-test export")
    if man["upstream_commit"] != PIN or man["sources"] != {
        k: digest(p) for k, p in sources(data).items()
    }:
        raise ValueError("dataset or version differs; recreate the export")
    for file, expected in man["files"].items():
        if digest(folder / file) != expected:
            raise ValueError("export digest mismatch: " + file)
    return man


def pack(folder, data, out):

    manifest = verify(folder, data)
    hashes = {**manifest["files"], "manifest.json": digest(folder / "manifest.json")}
    if out.exists():
        raise FileExistsError(f"use a new ZIP path: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        archive = Path(tmp) / "honeybee.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for name in sorted(hashes):
                if Path(name).name != name:
                    raise ValueError("unexpected export member name")
                entry = zipfile.ZipInfo(
                    "honeybee_export/" + name, (1980, 1, 1, 0, 0, 0)
                )
                entry.create_system = 3
                entry.external_attr = 0o100644 << 16
                z.writestr(
                    entry,
                    (folder / name).read_bytes(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=5,
                )
        with zipfile.ZipFile(archive) as z:
            for name, expected in hashes.items():
                if (
                    hashlib.sha256(z.read("honeybee_export/" + name)).hexdigest()
                    != expected
                ):
                    raise ValueError(f"ZIP read-back mismatch: {name}")
        archive.rename(out)
    print(f"{digest(out)}  {out.name}")


def checked_results(query, hits, facts, permissions, topk):
    if len(hits) > topk:
        raise ValueError("too many upstream results")
    evidence = []
    used = set()
    prev = -math.inf
    for bid, did, content, distance in hits:
        key = (int(bid), int(did))
        fact = facts.get(key)
        if (
            fact is None
            or fact["subject_id"] != query["subject_id"]
            or (query["user_id"], int(did)) not in permissions
            or key in used
        ):
            raise ValueError(
                "upstream returned a duplicate, foreign, unknown or forbidden block"
            )
        text = content if isinstance(content, str) else bytes(content).decode("utf-8")
        if (
            text != fact["text"]
            or not math.isfinite(float(distance))
            or float(distance) < prev
        ):
            raise ValueError("upstream block differs from export or distances unsorted")
        used.add(key)
        prev = float(distance)
        evidence.append(
            {
                "fact_id": fact["fact_id"],
                "doc_id": fact["doc_id"],
                "text": text,
                "distance": prev,
            }
        )
    seen = {f["fact_id"] for f in evidence}
    if any(set(j) <= seen for k in used for j in facts[k]["forbidden_joint_sets"] if j):
        raise ValueError(
            "upstream result violates joint rule; review the static RBAC mapping"
        )
    return evidence


def check_index(folder, facts, perms):

    from services.config import get_db_connection

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT role_id, document_id FROM PermissionAssignment")
            if set(cur.fetchall()) != perms:
                raise ValueError("DB permissions differ from export")
            users = {q["user_id"] for q in records(folder / "queries.jsonl")}
            cur.execute("SELECT user_id, role_id FROM UserRoles")
            if set(cur.fetchall()) != {(i, i) for i in users}:
                raise ValueError("DB principal roles differ from export")
            cur.execute(
                "SELECT block_id,document_id,block_content,vector::text FROM DocumentBlocks"
            )
            with (folder / "DocumentBlocks.csv").open(
                encoding="utf-8", newline=""
            ) as fp:
                vectors = {
                    (int(r["block_id"]), int(r["document_id"])): json.loads(r["vector"])
                    for r in csv.DictReader(fp)
                }
            found = set()
            for bid, did, text, vector_str in cur:
                k = (bid, did)
                if k not in facts or bytes(text).decode("utf-8") != facts[k]["text"]:
                    raise ValueError("DB source block differs from export")
                v = json.loads(vector_str)
                expected = vectors[k]
                if len(v) != len(expected) or any(
                    not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
                    for a, b in zip(v, expected)
                ):
                    raise ValueError("DB embedding differs from export")
                found.add(k)
            if found != set(facts):
                raise ValueError("DB missing document blocks")
            cur.execute("SELECT comb_role,partition_id FROM CombRolePartitions")
            mapping = {}
            for comb, part in cur.fetchall():
                if len(comb) == 1:
                    mapping.setdefault(comb[0], set()).add(part)
            required = {uid for uid, did in perms}
            if required - set(mapping):
                raise ValueError(
                    "missing upstream partition mapping for accessible roles"
                )
            cur.execute(
                "SELECT tablename FROM pg_indexes WHERE lower(indexdef) LIKE '%using hnsw%'"
            )
            indexed = {r[0] for r in cur.fetchall()}
            if any(
                "documentblocks_partition_" + str(p) not in indexed
                for u in required
                for p in mapping[u]
            ):
                raise ValueError(
                    "expected upstream HNSW indexes; check dynamic partition build"
                )
    finally:
        conn.close()


def query_index(folder, data, upstream, out, topk, ef_search):
    man = verify(folder, data)
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        commit != PIN
        or subprocess.run(
            [
                "git",
                "-C",
                str(upstream),
                "diff",
                "--quiet",
                "HEAD",
                "--",
                "controller",
                "services",
                "basic_benchmark",
            ]
        ).returncode
    ):
        raise ValueError("use upstream pinned commit with unmodified retrieval code")
    sys.path[:0] = [str(upstream), str(upstream / "basic_benchmark")]
    import efconfig

    efconfig.ef_search = ef_search
    from controller.dynamic_partition.search import dynamic_partition_search

    facts = {
        (r["db_block_id"], r["db_document_id"]): r
        for r in records(folder / "facts.jsonl")
    }
    with (folder / "PermissionAssignment.csv").open(encoding="utf-8", newline="") as fp:
        perms = {(int(r["role_id"]), int(r["document_id"])) for r in csv.DictReader(fp)}
    check_index(folder, facts, perms)
    enabled = {uid for uid, _ in perms}
    if out.exists():
        raise FileExistsError(f"use new output path: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        staged = Path(tmp) / "contexts.jsonl"
        with staged.open("w", encoding="utf-8") as fp:
            for q in records(folder / "queries.jsonl"):
                hits = []
                if q["user_id"] in enabled:
                    hits, _ = dynamic_partition_search(
                        q["user_id"],
                        q["query_vector"],
                        topk=topk,
                        statistics_type="sql",
                    )
                evidence = checked_results(q, hits, facts, perms, topk)
                row = {k: q[k] for k in ("request_id", "scenario_id", "subject_id")}
                if "authorization_digest" in q:
                    row["authorization_digest"] = q["authorization_digest"]
                row.update(
                    method="honeybee_external",
                    upstream_commit=PIN,
                    export_sha256=digest(folder / "manifest.json"),
                    workload_scope=man["scope"],
                    query_text=q["query_text"],
                    user_id=q["user_id"],
                    dataset_version=man["dataset_version"],
                    inputs=man["sources"],
                    evidence=evidence,
                    ef_search=ef_search,
                    topk=topk,
                )
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        staged.rename(out)
    print(f"Saved checked upstream retrieval contexts to {out}")


def main():
    p = argparse.ArgumentParser(
        description="Export a clinical RBAC workload or query a dedicated HoneyBee index."
    )
    sub = p.add_subparsers(dest="op", required=True)
    e = sub.add_parser("export")
    e.add_argument("--data", type=Path, default=Path("data"))
    e.add_argument(
        "--encoder", type=Path, default=Path(".cache/perm2ask/bge-small-en-v1.5")
    )
    e.add_argument("--limit", type=int)
    e.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("query")
    q.add_argument("--data", type=Path, default=Path("data"))
    q.add_argument("--export", type=Path, required=True)
    q.add_argument("--upstream", type=Path, required=True)
    q.add_argument("--topk", type=int, default=10)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--ef-search", type=int, default=40)
    z = sub.add_parser("pack")
    z.add_argument("--data", type=Path, default=Path("data"))
    z.add_argument("--export", type=Path, required=True)
    z.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.op == "export":
        if args.limit is not None and args.limit < 1:
            p.error("limit must be positive")
        export(args.data, args.encoder, args.out, args.limit)
    elif args.op == "pack":
        pack(args.export, args.data, args.out)
    else:
        if args.topk < 1 or args.ef_search < 1:
            p.error("topk and ef-search must be positive")
        query_index(
            args.export,
            args.data,
            args.upstream.resolve(),
            args.out,
            args.topk,
            args.ef_search,
        )


if __name__ == "__main__":
    main()
