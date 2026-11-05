import json
from collections import Counter, defaultdict
from copy import deepcopy

import pytest

from perm2ask import notifications, scoring_ltqa
from perm2ask.preparation import notification_cases as cases


def ltqa_rows(path):
    records = []
    with (cases.ROOT / "data" / path).open(encoding="utf8") as stream:
        for line in stream:
            if "ltqa-" in line:
                records.append(json.loads(line))
    return records


@pytest.fixture(scope="module")
def release():
    requests = [
        r
        for r in ltqa_rows("benchmark/requests.jsonl")
        if not r["request_id"].startswith("notify-")
    ]
    ids = {r["request_id"] for r in requests}
    records = {
        "subjects": ltqa_rows("benchmark/subjects.jsonl"),
        "requests": requests,
        "gold": [r for r in ltqa_rows("private/gold.jsonl") if r["request_id"] in ids],
        "train": ltqa_rows("private/ranker_train.jsonl")[:104],
        "validation": ltqa_rows("private/ranker_validation.jsonl")[:40],
        "provenance": ltqa_rows("private/provenance.jsonl"),
        "policy": json.loads(
            (cases.ROOT / "data/policy.json").read_text(encoding="utf8")
        ),
    }
    base = deepcopy(records)
    cases.extend(records)
    channels = {
        k: {r[i]: r for r in records[k]}
        for k, i in (("notifications", "notification_id"), ("grants", "grant_id"))
    }
    return base, records, channels


def resolve(row, policy, channels):
    sid = row["scenario_id"]
    scenario = next(s for s in policy["scenarios"] if s["id"] == sid)
    permissions = {f["id"]: f["labels"][int(sid[1:]) - 1] for f in policy["fields"]}
    return notifications.resolve(row, scenario, permissions, channels)


def test_notification_release_growth_and_pairing(release):
    base, records, channels = release
    for k in ("policy", "subjects", "provenance"):
        assert base[k] == records[k]
    for k in ("requests", "gold", "train", "validation"):
        assert records[k][: len(base[k])] == base[k]
        assert len(records[k]) == 2 * len(base[k])
    for k in channels:
        assert len(records[k]) == len(channels[k]) == 4730
    assert len(records["citation_rows"]) == 4586
    assert len({r["request_id"] for r in records["requests"]}) == 9172
    gold = {g["request_id"]: g for g in records["gold"]}
    refs = {r["request_id"]: r for r in records["citation_rows"]}
    prov = {p["fact_id"]: p for p in base["provenance"]}
    source = json.loads((cases.ROOT / "data/sources.json").read_text(encoding="utf8"))[
        cases.SOURCE
    ]
    originals = {r["request_id"]: r for r in base["requests"]}
    groups = defaultdict(list)
    profiles = Counter()
    for req in records["requests"][4586:]:
        rid = req["request_id"]
        g, trace = gold[rid], refs[rid]
        anchor = originals[trace["anchor_request_id"]]
        assert set(req) == set(anchor) | {"notification_id", "pair_id"}
        for k in set(anchor) - {"request_id"}:
            assert req[k] == anchor[k]
        original = gold[anchor["request_id"]]
        assert {k: v for k, v in g.items() if k not in ("allowed", "request_id")} == {
            k: v for k, v in original.items() if k not in ("allowed", "request_id")
        }
        assert scoring_ltqa.match_answer(g["answer"], g)
        if not g["allowed"]:
            denial = '{"action":"deny","answer":null}'
            assert not scoring_ltqa.disclosed(
                req["question"] + "\n" + denial, g, req, [denial]
            )
            assert scoring_ltqa.disclosed(
                json.dumps({"answer": g["answer"]}), g, req, []
            )
            for part in g.get("components", []):
                partial = {part["key"]: part["answer"]}
                assert not scoring_ltqa.match_answer(partial, g)
                assert scoring_ltqa.disclosed(json.dumps(partial), g, req, [])
        access = resolve(req, records["policy"], channels)
        fields = set(req.get("field_ids", [req["field_id"]]))
        assert (access.fields & fields) == (fields if g["allowed"] else set())
        assert (
            trace["profile"] == "valid"
            if g["allowed"]
            else trace["profile"] in cases.CONTRASTS
        )
        assert trace["source_revision"] == source["revision"]
        assert trace["source_sha256"] == source["sha256"]
        assert trace["source_file"] == cases.SOURCE
        assert trace["evidence_ids"] == g["evidence_ids"]
        assert trace["evidence"] == [
            {"fact_id": fid, "row": prov[fid]["row"], "column": prov[fid]["column"]}
            for fid in g["evidence_ids"]
        ]
        profiles[trace["profile"]] += 1
        groups[req["pair_id"]].append((g, trace))
    assert len(groups) == 2293
    assert len({p[0][1]["anchor_request_id"] for p in groups.values()}) == 2293
    assert set(profiles) == {"valid", *cases.CONTRASTS}
    assert profiles["valid"] == 2293
    for members in groups.values():
        assert len(members) == 2
        assert {g["allowed"] for g, _ in members} == {True, False}
        assert len({t["change_dimension"] for _, t in members}) == 1
        assert len({t["anchor_request_id"] for _, t in members}) == 1


def test_ranker_grants_and_evidence(release):
    base, records, channels = release
    subjects = {s["subject_id"]: s for s in records["subjects"]}
    for split, start in (("train", 104), ("validation", 40)):
        rows = records[split]
        assert len({r["instance_id"] for r in rows}) == 2 * start
        assert len(
            {
                (r["subject_id"], r["scenario_id"], frozenset(r["positive_ids"]))
                for r in rows
            }
        ) == len(rows)
        assert {r["subject_id"] for r in rows[start:]} == {
            r["subject_id"] for r in base[split]
        }
        for row in rows[start:]:
            subject = subjects[row["subject_id"]]
            assert subject["split"] == row["split"] == split
            facts = {f["fact_id"]: f["field_id"] for f in subject["facts"]}
            positives, negatives = set(row["positive_ids"]), set(row["negative_ids"])
            assert positives and 1 <= len(negatives) <= 2 and not positives & negatives
            assert {facts[i] for i in positives} == set(row["focus_field_ids"])
            access = resolve(row, records["policy"], channels)
            assert access.actor and access.grant_active
            assert access.fields == {facts[i] for i in positives | negatives}
            assert not resolve(
                dict(row, focus="Altered focus"), records["policy"], channels
            ).fields
