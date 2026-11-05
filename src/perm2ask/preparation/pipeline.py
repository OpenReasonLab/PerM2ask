import hashlib
import json
import shutil
import tempfile
import urllib.request
from collections import Counter
from copy import deepcopy
from pathlib import Path

from ..reference import EXPECTED_REQUESTS, VERSION
from . import (
    asclepius,
    ltqa,
    ltqa_extension,
    nhs,
    nhs_extension,
    notification_cases,
    supervision,
)
from .records import ROOT, build_records, sha

FILE_NAMES = {
    "subjects": "benchmark/subjects.jsonl",
    "requests": "benchmark/requests.jsonl",
    "gold": "private/gold.jsonl",
    "provenance": "private/provenance.jsonl",
    "train": "private/ranker_train.jsonl",
    "validation": "private/ranker_validation.jsonl",
    "notifications": "trusted/notifications.jsonl",
    "grants": "trusted/grants.jsonl",
    "citation_rows": "private/citations.jsonl",
}


def fetch_sources(raw):
    manifest = json.loads((ROOT / "data/sources.json").read_text(encoding="utf8"))
    for name, entry in manifest.items():
        target = raw / name
        if target.exists():
            if (
                target.stat().st_size != entry["bytes"]
                or sha(target) != entry["sha256"]
            ):
                raise ValueError(
                    f"Existing source differs from its pinned revision: {target}"
                )
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            download = Path(tmp) / "download"
            digest = hashlib.sha256()
            with (
                urllib.request.urlopen(entry["url"], timeout=180) as response,
                download.open("wb") as out,
            ):
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    digest.update(block)
                    out.write(block)
            if (
                download.stat().st_size != entry["bytes"]
                or digest.hexdigest() != entry["sha256"]
            ):
                raise ValueError(
                    f"Downloaded bytes differ from their pinned revision: {name}"
                )
            download.rename(target)


def save_files(folder, records):
    for key, name in FILE_NAMES.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf8", newline="\n") as stream:
            for row in records[key]:
                stream.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
    split_map = {s["subject_id"]: s["split"] for s in records["subjects"]}
    for name, obj in [("splits.json", split_map), ("policy.json", records["policy"])]:
        body = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
        (folder / name).write_bytes(body.replace("\n", "\r\n").encode("utf8"))


def expand(records):
    nhs_extension.extend(records)
    ltqa_extension.extend(records)
    notification_cases.extend(records)


def build_audit(folder, records):
    gold = {r["request_id"]: r["allowed"] for r in records["gold"]}
    if len(records["requests"]) != EXPECTED_REQUESTS or gold.keys() != {
        r["request_id"] for r in records["requests"]
    }:
        raise ValueError("Test references do not match the expected cohort")
    subjects = records["subjects"]
    labels = Counter((r["dataset"], gold[r["request_id"]]) for r in records["requests"])
    return {
        "version": VERSION,
        "base_commit": "5ea5e632d4ee3f28064fd9ca4da8974363e72021",
        "notification_test_pairs": len(
            {r["pair_id"] for r in records["citation_rows"]}
        ),
        "notification_test_profiles": dict(
            sorted(Counter(r["profile"] for r in records["citation_rows"]).items())
        ),
        "verified_facts": len(records["provenance"]),
        "request_count": len(records["requests"]),
        "subject_count": len(subjects),
        "test_subjects": sum(s["split"] == "test" for s in subjects),
        "counts": [
            {
                "dataset": ds,
                "split": "test",
                "subjects": sum(
                    s["split"] == "test" and s["dataset"] == ds for s in subjects
                ),
                "allowed": labels[ds, True],
                "denied": labels[ds, False],
            }
            for ds in ("asclepius", "nhs_synthetic", "ltqa")
        ],
        "ranker_splits": [
            {
                "split": split,
                "instances": len(records[split]),
                "subjects_with_valid_pairs": len(
                    {r["subject_id"] for r in records[split]}
                ),
                "by_dataset": dict(
                    sorted(Counter(r["dataset"] for r in records[split]).items())
                ),
            }
            for split in ("train", "validation")
        ],
        "sha256": {
            v: sha(folder / v)
            for v in [*FILE_NAMES.values(), "splits.json", "policy.json"]
        },
    }


def rebuild(raw, destination, fetch=False):
    if destination.exists():
        raise FileExistsError(f"Choose a new output directory: {destination}")
    if fetch:
        fetch_sources(raw)
    records = build_records(raw)
    initial_labels = {
        row["instance_id"]: deepcopy(row)
        for split in ("train", "validation")
        for row in records[split]
    }
    for source in (asclepius, nhs, ltqa):
        source.revise(records, raw)
    supervision.revise(records, initial_labels, ROOT / "configs/scenario_wordings.json")
    expand(records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        output = Path(tmp) / "benchmark"
        save_files(output, records)
        audit = json.loads((ROOT / "data/audit.json").read_text(encoding="utf8"))
        if build_audit(output, records) != audit:
            raise ValueError("Rebuilt records or counts differ from the release audit")
        for name in ("audit.json", "sources.json", "licenses/NHS-MIT.txt"):
            (output / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "data" / name, output / name)
        shutil.copyfile(ROOT / "DATA.md", output / "DATA.md")
        output.rename(destination)
