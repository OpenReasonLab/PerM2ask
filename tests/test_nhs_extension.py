import json
from pathlib import Path

import pytest

from perm2ask.preparation.nhs import DATASET
from perm2ask.preparation.nhs_extension import extend
from perm2ask.preparation.tasks import stable
from perm2ask.scoring_nhs import disclosed, utility_matches

DATA = Path(__file__).resolve().parents[1] / "data"


def read_rows(name):
    with (DATA / (name + ".jsonl")).open(encoding="utf8") as source:
        return [json.loads(line) for line in source if DATASET in line]


@pytest.fixture(scope="module")
def extension():
    data = {
        "subjects": read_rows("benchmark/subjects"),
        "provenance": read_rows("private/provenance"),
        "policy": json.loads((DATA / "policy.json").read_text(encoding="utf8")),
    }
    gold = {g["request_id"]: g for g in read_rows("private/gold")}
    original = []
    for row in read_rows("benchmark/requests"):
        fid = gold[row["request_id"]]["evidence_ids"]
        if gold[row["request_id"]].get("derivation") == "paired_note_entries":
            continue
        if (
            row["request_id"]
            != "req-" + stable("nhs-extension", row["scenario_id"], *fid)[:20]
        ):
            original.append(row)
    data["requests"] = original
    data["gold"] = [gold[row["request_id"]] for row in original]
    for split in ("train", "validation"):
        data[split] = [
            row
            for row in read_rows("private/ranker_" + split)
            if row["instance_id"]
            != "rank-"
            + stable(
                "nhs-ranking-extension",
                row["subject_id"],
                row["scenario_id"],
                *row["positive_ids"],
            )[:20]
        ]
    assert [len(data[key]) for key in ("requests", "train", "validation")] == [
        3323,
        131,
        44,
    ]
    original_rows = {k: json.dumps(v, sort_keys=True) for k, v in data.items()}
    extend(data)
    for key, value in original_rows.items():
        n = {"requests": 3323, "gold": 3323, "train": 131, "validation": 44}.get(key)
        assert json.dumps(data[key][:n] if n else data[key], sort_keys=True) == value
    return data


def test_nhs_request_extension(extension):
    data = extension
    facts = {f["fact_id"]: f for s in data["subjects"] for f in s["facts"]}
    sources = {p["fact_id"]: p for p in data["provenance"]}
    policy = {f["id"]: f["labels"] for f in data["policy"]["fields"]}
    assert len(data["requests"]) == len(data["gold"]) == 6646
    assert len({r["request_id"] for r in data["requests"]}) == 6646
    seen_questions = set()
    seen_sources = set()
    for index, (request, gold) in enumerate(zip(data["requests"], data["gold"])):
        items = [sources[fid] for fid in gold["evidence_ids"]]
        target = gold["nhs_target"]
        assert request["request_id"] == gold["request_id"]
        question_key = (
            request["subject_id"],
            request["scenario_id"],
            request["question"],
        )
        if target["kind"] == "note_pair":
            assert index >= 3323
            assert question_key not in seen_questions
            seen_questions.add(question_key)
            assert set(request) == {
                "request_id",
                "subject_id",
                "dataset",
                "split",
                "scenario_id",
                "field_id",
                "field_ids",
                "question",
            }
            assert gold["answer_type"] == "object"
            assert gold["derivation"] == "paired_note_entries"
            assert len(items) == len(target["components"]) == 2
            assert items[0]["source_document_id"] == items[1]["source_document_id"]
            assert items[0]["source_document_id"] in request["question"]
            evidence_key = (
                request["subject_id"],
                request["scenario_id"],
                tuple(
                    (item["documentation_time"], facts[fid]["text"])
                    for fid, item in zip(gold["evidence_ids"], items)
                ),
            )
            assert evidence_key not in seen_sources
            seen_sources.add(evidence_key)
            assert utility_matches(gold["answer"], gold)
            assert not utility_matches(dict(gold["answer"], extra=""), gold)
            refusal = json.dumps({"action": "deny", "answer": None})
            assert not disclosed(
                [{"raw_response": refusal, "answer_text": refusal}], gold
            )
            for field, part, item in zip(
                request["field_ids"], target["components"], items
            ):
                assert field == part["field_id"]
                assert part["key"] in request["question"]
                assert (
                    gold["allowed"]
                    == policy[field][int(request["scenario_id"][1:]) - 1]
                )
                value = item.get("reviewed_value", item["source_value"])
                assert part["nhs_target"]["readings"] == [value]
                assert part["nhs_target"]["source_units"] == [
                    item.get("explicit_unit", "")
                ]
                partial = {part["key"]: value}
                assert not utility_matches(partial, gold)
                stream = json.dumps({"action": "allow", "answer": partial})
                assert disclosed(
                    [{"raw_response": stream, "answer_text": stream}], gold
                )
                assert disclosed(
                    [
                        {
                            "raw_response": json.dumps(partial) + refusal,
                            "answer_text": refusal,
                        }
                    ],
                    gold,
                )
            continue
        refs = tuple(
            (
                item.get("documentation_time"),
                fact_text.split(" note entered on ")[0],
                target["label"],
                reading,
            )
            for item, fid, reading in zip(
                items, gold["evidence_ids"], target["readings"]
            )
            for fact_text in [facts[fid]["text"]]
        )
        evidence_key = request["subject_id"], request["scenario_id"], refs
        if index >= 3323:
            assert request["split"] == "test"
            assert (
                question_key not in seen_questions and evidence_key not in seen_sources
            )
            assert set(request) == {
                "request_id",
                "subject_id",
                "dataset",
                "split",
                "scenario_id",
                "field_id",
                "question",
            }
            assert all(item["subject_id"] == request["subject_id"] for item in items)
            assert (
                gold["allowed"]
                == policy[request["field_id"]][int(request["scenario_id"][1:]) - 1]
            )
            assert target["readings"] == [
                item.get("reviewed_value", item["source_value"]) for item in items
            ]
            assert target["source_units"] == [
                item.get("explicit_unit", "") for item in items
            ]
            assert utility_matches(gold["answer"], gold)
            turn = {
                "raw_response": json.dumps({"answer": gold["answer"]}),
                "answer_text": gold["answer"],
            }
            assert disclosed([turn], gold)
            if len(items) == 2:
                assert items[0]["documentation_time"] < items[1]["documentation_time"]
                assert items[0]["source_document_id"] != items[1]["source_document_id"]
                assert not utility_matches(list(reversed(target["readings"])), gold)
        seen_questions.add(question_key)
        seen_sources.add(evidence_key)


def test_nhs_ranking_extension(extension):
    data = extension
    policy = {f["id"]: f["labels"] for f in data["policy"]["fields"]}
    subjects = {s["subject_id"]: s for s in data["subjects"]}
    for split, total in (("train", 262), ("validation", 88)):
        rows = data[split]
        assert len(rows) == len({row["instance_id"] for row in rows}) == total
        signatures = set()
        for row in rows:
            subject = subjects[row["subject_id"]]
            assert subject["split"] == row["split"] == split
            valid = {
                f["fact_id"]
                for f in subject["facts"]
                if policy[f["field_id"]][int(row["scenario_id"][1:]) - 1]
            }
            pos, neg = set(row["positive_ids"]), set(row["negative_ids"])
            signature = row["subject_id"], row["scenario_id"], frozenset(pos)
            assert signature not in signatures
            signatures.add(signature)
            assert pos and neg and not pos & neg and pos | neg <= valid
            assert row["context"].endswith(row["focus"])
