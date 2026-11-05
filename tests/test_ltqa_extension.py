import json
from collections import defaultdict
from copy import deepcopy

import pytest

from perm2ask import scoring_ltqa
from perm2ask.preparation import ltqa_extension as extension


def ltqa_rows(path):
    lines = (extension.ROOT / "data" / path).read_text(encoding="utf8").splitlines()
    return [json.loads(line) for line in lines if "ltqa-" in line]


@pytest.fixture(scope="module")
def release():
    requests = ltqa_rows("benchmark/requests.jsonl")[:2293]
    ids = {r["request_id"] for r in requests}
    data = {
        "policy": json.loads(
            (extension.ROOT / "data/policy.json").read_text(encoding="utf8")
        ),
        "subjects": ltqa_rows("benchmark/subjects.jsonl"),
        "provenance": ltqa_rows("private/provenance.jsonl"),
        "requests": requests,
        "gold": [g for g in ltqa_rows("private/gold.jsonl") if g["request_id"] in ids],
        "train": ltqa_rows("private/ranker_train.jsonl")[:52],
        "validation": ltqa_rows("private/ranker_validation.jsonl")[:20],
    }
    original = deepcopy(data)
    extension.extend(data)
    return original, data


def test_preserves_base_and_appends_grounded_abstracts(release):
    old, data = release
    for key in ("policy", "subjects", "provenance"):
        assert data[key] == old[key]
    for key in ("requests", "gold", "train", "validation"):
        assert data[key][: len(old[key])] == old[key]
        assert len(data[key]) == 2 * len(old[key])
    assert len({r["request_id"] for r in data["requests"]}) == 4586
    sources = {p["fact_id"]: p for p in data["provenance"]}
    policies = {f["id"]: f["labels"] for f in data["policy"]["fields"]}
    prior = {
        (r["subject_id"], r["scenario_id"], frozenset(r["field_ids"]))
        for r in old["requests"]
        if "field_ids" in r
    }
    observations = defaultdict(set)
    held_out = {s["subject_id"] for s in data["subjects"] if s["split"] == "test"}
    new_requests = data["requests"][2293:]
    assert {r["subject_id"] for r in new_requests} == held_out
    for r, g in zip(new_requests, data["gold"][2293:]):
        uid, sid = r["subject_id"], r["scenario_id"]
        fields = tuple(r["field_ids"])
        key = uid, sid, frozenset(fields)
        assert key not in prior
        prior.add(key)
        assert len(set(fields)) == len(fields) in (2, 3)
        assert {policies[f][int(sid[1:]) - 1] for f in fields} == {g["allowed"]}
        assert r["request_id"] == g["request_id"]
        assert scoring_ltqa.match_answer(g["answer"], g)
        for field, fact_id, part in zip(fields, g["evidence_ids"], g["components"]):
            source = sources[fact_id]
            assert source["subject_id"] == uid
            assert source["column"] == "profile." + extension.FIELDS[field][0]
            assert part["answer"] == source.get(
                "reviewed_value", source["source_value"]
            )
            assert part["evidence_ids"] == [fact_id]
            values = (
                part["answer"] if part["answer_type"] == "list" else [part["answer"]]
            )
            assert not any(
                scoring_ltqa.token(scoring_ltqa.normal(r["question"]), value)
                for value in values
            )
        observations[sid, fields].add(json.dumps(g["answer"], sort_keys=True))
    assert all(len(values) >= 2 for values in observations.values())


def test_ranker_partitions_have_new_authorized_evidence_sets(release):
    old, data = release
    subjects = {s["subject_id"]: s for s in data["subjects"]}
    policies = {f["id"]: f["labels"] for f in data["policy"]["fields"]}
    for split, count in (("train", 52), ("validation", 20)):
        rows = data[split]
        signatures = {
            (r["subject_id"], r["scenario_id"], frozenset(r["positive_ids"]))
            for r in rows
        }
        assert len(signatures) == len({r["instance_id"] for r in rows}) == 2 * count
        assert {r["subject_id"] for r in rows[count:]} == {
            r["subject_id"] for r in old[split]
        }
        for row in rows[count:]:
            s = subjects[row["subject_id"]]
            assert s["split"] == row["split"] == split
            eligible = {
                f["fact_id"]: f["field_id"]
                for f in s["facts"]
                if policies[f["field_id"]][int(row["scenario_id"][1:]) - 1]
            }
            pos, neg = set(row["positive_ids"]), set(row["negative_ids"])
            assert pos and neg and pos.isdisjoint(neg)
            assert pos | neg == eligible.keys()
            assert {eligible[i] for i in pos} == set(row["focus_field_ids"])


def test_partial_disclosure_and_typed_utility(release):
    _, data = release
    covered = set()
    denial = '{"action":"deny","answer":null}'
    for r, g in zip(data["requests"][2293:], data["gold"][2293:]):
        if g["allowed"]:
            continue
        for quote in ("", r["question"], json.dumps({"question": r["question"]})):
            assert not scoring_ltqa.disclosed(quote + "\n" + denial, g, r, [denial])
        for p in g["components"]:
            key, value = p["key"], p["answer"]
            covered.add((p["answer_type"], json.dumps(value)))
            partial = {key: value}
            assert not scoring_ltqa.match_answer(partial, g)
            invalid = dict(g["answer"], **{key: {}})
            assert not scoring_ltqa.match_answer(invalid, g)
            fragments = [
                json.dumps(partial),
                json.dumps({"action": "deny", "answer": partial}),
                key + ": " + json.dumps(value),
            ]
            if p["answer_type"] == "list":
                fragments.extend(json.dumps({key: [entry]}) for entry in value)
            if p["answer_type"] == "boolean":
                fragments.append(key + ": " + ("yes" if value else "no"))
            for fragment in fragments:
                assert scoring_ltqa.disclosed(fragment + "\n" + denial, g, r, [denial])
    assert {("list", "[]"), ("boolean", "true"), ("boolean", "false")} <= covered
