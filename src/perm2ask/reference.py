import hashlib
import json
from pathlib import Path

VERSION = "independent-reconstruction-test26218-v5-notifications"
DATASETS = {"asclepius", "nhs_synthetic", "ltqa"}
EXPECTED_TEST = {"asclepius": 10400, "nhs_synthetic": 6646, "ltqa": 9172}
EXPECTED_REQUESTS = sum(EXPECTED_TEST.values())
EXPECTED_RANKER = {"train": 1166, "validation": 281}


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_audit(path):
    audit = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if audit["version"] != VERSION:
        raise ValueError(
            f"wrong benchmark version: expected {VERSION}, got {audit['version']}"
        )
    counts = {}
    for row in audit["counts"]:
        if row["split"] != "test":
            continue
        name = row["dataset"]
        if name in counts or any(
            type(row[k]) is not int or row[k] <= 0 for k in ("allowed", "denied")
        ):
            raise ValueError("invalid per-dataset final test denominator")
        counts[name] = {"allowed": row["allowed"], "denied": row["denied"]}
    if (
        set(counts) != DATASETS
        or {k: sum(v.values()) for k, v in counts.items()} != EXPECTED_TEST
    ):
        raise ValueError(
            "three dataset denominators must sum to the expanded final-test total"
        )
    if audit["request_count"] != EXPECTED_REQUESTS:
        raise ValueError(
            "request count differs from the expected expanded benchmark size"
        )
    ranker_counts = {r["split"]: r["instances"] for r in audit["ranker_splits"]}
    if ranker_counts != EXPECTED_RANKER:
        raise ValueError("unexpected auxiliary train/validation size")
    return audit, counts
