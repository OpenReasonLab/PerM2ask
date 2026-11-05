import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def summarize(path):
    predictions = defaultdict(dict)
    cohort = {}
    counts = Counter()
    letters = set("ABCDEFGHIJ")
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            key = (row["model"], row["round"])
            trial = row["trial_id"]
            label, pred = row["gold_letter"], row["predicted_letter"]
            if (
                trial in predictions[key]
                or label not in letters
                or pred not in letters | {None}
            ):
                raise ValueError(f"Invalid or duplicate trial {trial} in {key}")
            entry = row["source_key"], row["dataset"], label, row["dataset_ordinal"]
            if cohort.setdefault(trial, entry) != entry:
                raise ValueError(f"Cohort mismatch for {trial}")
            predictions[key][trial] = pred
            counts[key, label] += 1
    expected = {(m, r) for m in ("Qwen3.5-0.8B", "Qwen3.5-2B") for r in range(1, 4)}
    if set(predictions) != expected or len(cohort) != 5000:
        raise ValueError("Incomplete rounds or cohort")
    summary = []
    for model, repeat in sorted(expected):
        key = model, repeat
        trial_preds = predictions[key]
        if trial_preds.keys() != cohort.keys() or any(
            counts[key, label] != 500 for label in letters
        ):
            raise ValueError(f"Missing trials in {key}")
        correct = sum(p == cohort[trial][2] for trial, p in trial_preds.items())
        summary.append(
            {
                "model": model,
                "round": repeat,
                "calls": len(cohort),
                "correct": correct,
                "accuracy_percent": 100 * correct / len(cohort),
                "changed_since_first_round": sum(
                    p != predictions[model, 1][trial]
                    for trial, p in trial_preds.items()
                ),
            }
        )
    return {"distinct_sources": len({r[0] for r in cohort.values()}), "rounds": summary}


def main():
    parser = argparse.ArgumentParser(
        description="Score the saved A-J image-type predictions."
    )
    parser.add_argument(
        "--predictions", type=Path, default=Path("data/image_types/predictions.jsonl")
    )
    args = parser.parse_args()
    print(json.dumps(summarize(args.predictions), indent=2))


if __name__ == "__main__":
    main()
