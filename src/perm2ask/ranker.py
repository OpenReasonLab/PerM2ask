from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F

from . import notifications
from .benchmark import Benchmark, read_rows
from .method import (
    EvidenceSelector,
    FrozenBGE,
    RankNet,
    admissible_pack,
    digest,
    encode_view,
)


def lineage(root):
    files = (
        "benchmark/subjects.jsonl",
        "policy.json",
        "splits.json",
        *notifications.FILES.values(),
    )
    return {name: digest(root / name) for name in files if (root / name).is_file()}


def tokenizer_counter(path):
    from tokenizers import Tokenizer

    path = Path(path)
    if path.is_dir():
        path = path / "tokenizer.json"
    tok = Tokenizer.from_file(str(path))
    return lambda text: len(tok.encode(text, add_special_tokens=False).ids)


def eligible_rows(db, rows, split, limit):
    eligible = []
    keys = set()
    for row in rows:
        if row.get("split") != split:
            continue
        instance_id = row["instance_id"]
        if not instance_id or instance_id in keys:
            raise ValueError(f"Empty or duplicate training instance ID: {instance_id}")
        keys.add(instance_id)
        marked = notifications.notification_case(row)
        if marked:
            case = db.case_for(row)
        else:
            case = db.case(row["subject_id"], row["scenario_id"])
        if not isinstance(row["context"], str) or not row["context"].strip():
            raise ValueError(
                "Training row requires an explicit trusted rewritten context"
            )
        if not marked:
            case["context"] = row["context"]
        if case["split"] != split:
            raise ValueError("Label partition does not match subject partition")
        pos, neg = row["positive_ids"], row["negative_ids"]
        if len(set(pos)) != len(pos) or len(set(neg)) != len(neg):
            raise ValueError("Duplicate annotated ID")
        if set(pos) & set(neg) or not (set(pos) | set(neg)) <= set(
            case["authorized_ids"]
        ):
            raise ValueError("Evidence labels must be disjoint authorized subsets")
        if pos and neg:
            eligible.append((case, row))
            if limit and len(eligible) >= limit:
                break
    if not eligible:
        raise ValueError(f"No eligible labeled instances in {split}")
    return eligible


def prepare(entries, encoder, k):
    views = []
    for case, row in entries:
        view = encode_view(case, encoder, k)
        order = {f["id"]: i for i, f in enumerate(view[0])}
        p = [order[fid] for fid in row["positive_ids"]]
        n = [order[fid] for fid in row["negative_ids"]]
        views.append((case, view, p, n))
    return views


@torch.no_grad()
def validation(net, rows, counter, budget):
    recall, coverage, ndcg = [], [], []
    for case, view, pos, _ in rows:
        scores = net(view)
        chosen = admissible_pack(case, view[0], scores, counter, budget)
        positive = {view[0][i]["id"] for i in pos}
        hits = len(positive & {f["fact_id"] for f in chosen})
        recall.append(hits / len(positive))
        coverage.append(int(hits == len(positive)))
        ranks = torch.argsort(scores, descending=True, stable=True).tolist()
        dcg = sum(1 / math.log2(j + 2) for j, i in enumerate(ranks) if i in pos)
        ideal = sum(1 / math.log2(j + 2) for j in range(len(pos)))
        ndcg.append(dcg / ideal)
    return {
        "recall": sum(recall) / len(rows),
        "complete": sum(coverage) / len(rows),
        "ndcg": sum(ndcg) / len(rows),
    }


def train(args, db, encoder, count):
    if args.method == "without_hgnn":
        raise ValueError("without_hgnn has no training")
    if not args.labels and not args.weak_labels:
        raise ValueError(
            "Provide reviewed --labels JSONL or explicitly select reconstruction --weak-labels"
        )
    train_rows, val_rows = [], []
    label_hashes = {}
    for split, target in (("train", train_rows), ("validation", val_rows)):
        label_path = (
            args.datasets / f"private/ranker_{split}.jsonl"
            if args.weak_labels
            else args.labels
        )
        if label_path.is_dir():
            label_path = label_path / f"ranker_{split}.jsonl"
        label_hashes[split] = digest(label_path)
        rows = read_rows(label_path)
        target.extend(eligible_rows(db, rows, split, args.limit))
    training_subjects = sorted({case["subject_id"] for case, _ in train_rows})
    validation_subjects = sorted({case["subject_id"] for case, _ in val_rows})
    if set(training_subjects) & set(validation_subjects):
        raise ValueError("Subject overlap across train and validation")
    train_data = prepare(train_rows, encoder, args.k)
    val_data = prepare(val_rows, encoder, args.k)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    net = RankNet(args.method, encoder.dimension, args.hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    best = None
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_data)
        net.train()
        loss_total = 0.0
        for start in range(0, len(train_data), args.batch_size):
            losses = []
            for _, view, pos, neg in train_data[start : start + args.batch_size]:
                scores = net(view)
                losses.append(
                    F.softplus(scores[neg][None, :] - scores[pos][:, None]).mean()
                )
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_total += sum(float(item.detach()) for item in losses)
        net.eval()
        metrics = validation(net, val_data, count, args.budget)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "training_loss": loss_total / len(train_data),
                    "validation": metrics,
                }
            ),
            flush=True,
        )
        criteria = tuple(metrics[key] for key in ("recall", "complete", "ndcg"))
        if best is None or criteria > best[0]:
            state = {
                key: value.detach().clone() for key, value in net.state_dict().items()
            }
            best = (criteria, epoch, metrics, state)
    ck = {
        "method": args.method,
        "encoder_signature": encoder.signature,
        "k": args.k,
        "budget": args.budget,
        "dimension": encoder.dimension,
        "hidden": args.hidden,
        "training_subjects": training_subjects,
        "validation_subjects": validation_subjects,
        "epoch": best[1],
        "validation": best[2],
        "state": best[3],
        "supervision": "reviewed_reconstructed_weak_labels"
        if args.weak_labels
        else "user_supplied",
        "label_files_sha256": label_hashes,
        "dataset_sha256": lineage(args.datasets),
        "ranker_code_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in ("ranker.py", "benchmark.py", "method.py", "notifications.py")
        },
        "training_tokenizer_sha256": digest(
            args.tokenizer / "tokenizer.json"
            if args.tokenizer.is_dir()
            else args.tokenizer
        ),
        "n_train": len(train_data),
        "n_validation": len(val_data),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "lr": args.lr,
        "epochs": args.epochs,
        "limit": args.limit,
    }
    torch.save(ck, args.tmp)


def select(args, db, encoder, count):
    selector = EvidenceSelector(
        args.method, encoder, args.checkpoint, args.k, args.budget
    )
    targets = None
    if args.request_ids:
        lines = [
            line.strip()
            for line in args.request_ids.read_text(encoding="utf-8-sig").splitlines()
        ]
        if (
            not lines
            or any(not line for line in lines)
            or len(lines) != len(set(lines))
        ):
            raise ValueError(
                "Expected a nonempty UTF-8 list of unique request IDs, one per line"
            )
        targets = set(lines)
    cache = {}
    selected = 0
    ids = set()
    config = {
        "encoder": encoder.signature,
        "k": args.k,
        "budget": args.budget,
        "tokenizer_sha256": digest(
            args.tokenizer / "tokenizer.json"
            if args.tokenizer.is_dir()
            else args.tokenizer
        ),
        "checkpoint_sha256": digest(args.checkpoint) if args.checkpoint else None,
        "request_ids_sha256": digest(args.request_ids) if args.request_ids else None,
        "subjects_sha256": digest(args.datasets / "benchmark/subjects.jsonl"),
        "requests_sha256": digest(db.requests),
        "policy_sha256": digest(args.datasets / "policy.json"),
    }
    for name, relative in notifications.FILES.items():
        path = args.datasets / relative
        if path.is_file():
            config[name + "_sha256"] = digest(path)
    with args.tmp.open("w", encoding="utf-8") as out:
        for request in read_rows(db.requests):
            if request["split"] != args.split:
                continue
            rid = request["request_id"]
            if targets is not None and rid not in targets:
                continue
            key = (request["subject_id"], request["scenario_id"])
            case = None
            if notifications.notification_case(request):
                case = db.case_for(request)
                key += (case["authorization_digest"],)
            if key not in cache:
                if case is None:
                    case = db.case_for(request)
                cache[key] = selector.select(case, count)
            if rid in ids:
                raise ValueError(f"Duplicate request ID {rid}")
            ids.add(rid)
            row = {
                "request_id": rid,
                "method": args.method,
                "evidence": cache[key],
                "ranker_config": config,
            }
            if notifications.notification_case(request):
                row["authorization_digest"] = case["authorization_digest"]
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            selected += 1
            if args.limit and selected >= args.limit:
                break
    if targets is not None and ids != targets:
        missing = sorted(targets - ids)
        raise ValueError(
            f"{len(missing)} requested ID(s) absent from {args.split}: {missing[:5]}"
        )
    if not selected:
        raise ValueError("Selected partition has no requests")
    print(f"Selected authorized evidence for {selected} requests")


def main():
    p = argparse.ArgumentParser(
        description="Train and query the permission-gated evidence ranker."
    )
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("train", "select"):
        child = sub.add_parser(name)
        child.add_argument("--datasets", type=Path, default=Path("data"))
        child.add_argument(
            "--method", choices=["full", "pg", "without_hgnn"], required=True
        )
        child.add_argument(
            "--encoder", type=Path, default=Path(".cache/perm2ask/bge-small-en-v1.5")
        )
        child.add_argument(
            "--tokenizer",
            type=Path,
            required=True,
            help="Generation tokenizer.json or its folder",
        )
        child.add_argument("--k", type=int, default=5)
        child.add_argument("--budget", type=int, default=2048)
        child.add_argument("--out", type=Path, required=True)
        if name == "select":
            selection = child.add_mutually_exclusive_group()
            selection.add_argument("--limit", type=int)
            selection.add_argument(
                "--request-ids",
                type=Path,
                help="UTF-8 file with one unique request ID per line",
            )
            child.add_argument("--checkpoint", type=Path)
            child.add_argument(
                "--split", choices=["train", "validation", "test"], default="test"
            )
        else:
            child.add_argument("--limit", type=int)
            group = child.add_mutually_exclusive_group(required=True)
            group.add_argument("--labels", type=Path)
            group.add_argument("--weak-labels", action="store_true")
            child.add_argument("--hidden", type=int, default=64)
            child.add_argument("--epochs", type=int, default=5)
            child.add_argument("--batch-size", type=int, default=8)
            child.add_argument("--lr", type=float, default=1e-3)
            child.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    for key in ("budget", "k", "limit", "epochs", "hidden", "batch_size", "lr"):
        value = getattr(args, key, None)
        if value is not None and value <= 0:
            p.error(f"{key} must be positive")
    torch.set_num_threads(2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        p.error("Output exists. Choose a new output path.")
    with tempfile.NamedTemporaryFile(delete=False, dir=args.out.parent) as tmp:
        args.tmp = Path(tmp.name)
    try:
        db = Benchmark(args.datasets)
        encoder = FrozenBGE(args.encoder)
        counter = tokenizer_counter(args.tokenizer)
        (train if args.command == "train" else select)(args, db, encoder, counter)

        os.link(args.tmp, args.out)
    finally:
        args.tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
