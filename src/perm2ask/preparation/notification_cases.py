import json
from collections import defaultdict
from copy import deepcopy
from itertools import islice

from . import ltqa_extension
from .records import ROOT, stable
from .sampling import round_robin


SOURCE = "ltqa/airgapagent-r.json"
PAIR_COUNT = 2293

RANK_TASKS = {
    "S1": "For the appointment booking, identify the recorded registration or contact entries for",
    "S2": "For the laboratory follow-up, locate the intake history entries documenting",
    "S3": "For medication reconciliation, retrieve the source entries that document",
    "S4": "For the pre-vaccination review, locate the intake entries that report",
    "S5": "For the nutrition follow-up, assemble the intake history entries for",
    "S6": "For the pre-anesthesia assessment, identify the prior intake entries reporting",
    "S7": "For the rehabilitation visit, locate the intake history entries for",
    "S8": "For the behavioral health follow-up, find the intake entries that document",
    "S9": "For the emergency review, find the available patient-reported entries for",
}


def approved_anchors(records, gold):
    rows = [r for r in records["requests"] if r["dataset"] == "ltqa"]
    if len(rows) != 4586:
        raise ValueError("Expected LTQA v4 requests before adding notification cases")
    legacy = []
    pools = defaultdict(list)
    for req in rows:
        reference = gold[req["request_id"]]
        if not reference["allowed"]:
            continue
        if reference.get("derivation") == "intake_abstract":
            pools[req["subject_id"], req["scenario_id"]].append(req)
        else:
            legacy.append(req)
    if len(legacy) != 1210:
        raise ValueError("Unexpected number of policy-allowed LTQA v3 anchors")
    for pool in pools.values():
        pool.sort(key=lambda r: stable("notification-anchor", r["request_id"]))
    abstracts = round_robin([pools[key] for key in sorted(pools, key=stable)])
    anchors = legacy + list(islice(abstracts, PAIR_COUNT - len(legacy)))
    signatures = {(r["subject_id"], r["scenario_id"], r["question"]) for r in anchors}
    if len(signatures) != PAIR_COUNT:
        raise ValueError("Expected distinct patient, scenario and question anchors")
    return sorted(
        anchors, key=lambda r: stable("paired-notifications", r["request_id"])
    )


def retrieval_rows(records):
    subjects = {
        s["subject_id"]: s
        for s in records["subjects"]
        if s["dataset"] == "ltqa" and s["split"] != "test"
    }
    labels = {f["id"]: f["labels"] for f in records["policy"]["fields"]}
    variants = defaultdict(list)
    for row in json.loads(
        (ROOT / "configs/scenario_wordings.json").read_text(encoding="utf8")
    ):
        variants[row["split"], row["scenario_id"]].append(row)
    groups = ltqa_extension.catalog()
    result = {}
    for split, target in (("train", 104), ("validation", 40)):
        if sum(r["dataset"] == "ltqa" for r in records[split]) != target:
            raise ValueError(f"Expected the v4 LTQA {split} partition")
        rows = ltqa_extension.ranker_rows(
            records, subjects, groups, labels, split, target, variants
        )
        for row in rows:
            fields = row["focus_field_ids"]
            names = ltqa_extension.joined([ltqa_extension.LABELS[f] for f in fields])
            focus = f"{RANK_TASKS[row['scenario_id']]} {names}."
            row["context"] = row["context"].removesuffix(row["focus"]) + focus
            row["focus"] = focus
        result[split] = rows
    return result


def citation(
    anchor, request, reference, grant_id, profile, dimension, provenance, source
):
    evidence = []
    for fid in reference["evidence_ids"]:
        entry = provenance[fid]
        if entry["file"] != SOURCE or entry["subject_id"] != request["subject_id"]:
            raise ValueError(f"LTQA source mismatch for {fid}")
        evidence.append(dict(fact_id=fid, row=entry["row"], column=entry["column"]))
    return dict(
        request_id=request["request_id"],
        anchor_request_id=anchor["request_id"],
        notification_id=request["notification_id"],
        pair_id=request["pair_id"],
        grant_id=grant_id,
        profile=profile,
        change_dimension=dimension,
        source_file=SOURCE,
        source_revision=source["revision"],
        source_sha256=source["sha256"],
        evidence_ids=list(reference["evidence_ids"]),
        evidence=evidence,
    )


CONTRASTS = {
    "wrong_actor": "notification_actor",
    "bad_signature": "notification_signature",
    "expired_notification": "notification_validity",
    "revoked_grant": "grant_status",
    "narrow_grant": "grant_scope",
    "wrong_grant_subject": "grant_subject",
}


def extend(records):
    from ..notifications import fixture_channels

    gold = {g["request_id"]: g for g in records["gold"]}
    anchors = approved_anchors(records, gold)
    ranks = retrieval_rows(records)
    provenance = {p["fact_id"]: p for p in records["provenance"] if p["file"] == SOURCE}
    source = json.loads((ROOT / "data/sources.json").read_text(encoding="utf8"))[SOURCE]
    scenarios = {s["id"]: s for s in records["policy"]["scenarios"]}
    policies = {f["id"]: f["labels"] for f in records["policy"]["fields"]}
    contrasts = tuple(CONTRASTS)
    staged = []
    for index, anchor in enumerate(anchors):
        rid = anchor["request_id"]
        pair_id = "pair-" + stable("notification-pair", rid)[:20]
        scenario = scenarios[anchor["scenario_id"]]
        fields = anchor.get("field_ids", [anchor["field_id"]])
        column = int(scenario["id"][1:]) - 1
        if not all(policies[f][column] for f in fields):
            raise ValueError(f"LTQA anchor exceeds its trusted scenario: {rid}")
        contrast = contrasts[index % len(contrasts)]
        for profile in ("valid", contrast):
            request = deepcopy(anchor)
            request["request_id"] = (
                "notify-" + stable("paired-request", rid, profile)[:20]
            )
            request["pair_id"] = pair_id
            request["notification_id"] = (
                "event-" + stable("paired-event", request["request_id"])[:20]
            )
            grant_fields = fields
            if profile == "narrow_grant":
                grant_fields = [
                    f
                    for f in ltqa_extension.FIELDS
                    if policies[f][column] and f not in fields
                ][: len(fields)]
            notification, grant = fixture_channels(
                request, scenario, grant_fields, profile
            )
            reference = deepcopy(gold[rid])
            reference.update(
                request_id=request["request_id"], allowed=profile == "valid"
            )
            trace = citation(
                anchor,
                request,
                reference,
                grant["grant_id"],
                profile,
                CONTRASTS[contrast],
                provenance,
                source,
            )
            staged.append((request, reference, notification, grant, trace))
    rank_events = []
    non_test_facts = {
        f["fact_id"]: f["field_id"]
        for s in records["subjects"]
        if s["dataset"] == "ltqa" and s["split"] != "test"
        for f in s["facts"]
    }
    for rows in ranks.values():
        for row in rows:
            iid = "notify-rank-" + stable("signed-retrieval", row["instance_id"])[:20]
            row.update(
                instance_id=iid,
                pair_id="rank-focus-" + stable("signed-focus", iid)[:20],
                notification_id="event-" + stable("signed-rank-event", iid)[:20],
            )
            row["negative_ids"] = sorted(
                row["negative_ids"], key=lambda fid: stable(iid, fid)
            )[:2]
            field_ids = sorted(
                {
                    non_test_facts[fid]
                    for fid in row["positive_ids"] + row["negative_ids"]
                }
            )
            rank_events.append(
                fixture_channels(row, scenarios[row["scenario_id"]], field_ids)
            )
    for row in sorted(staged, key=lambda row: row[0]["request_id"]):
        for key, item in zip(
            ("requests", "gold", "notifications", "grants", "citation_rows"), row
        ):
            records.setdefault(key, []).append(item)
    for notice, grant in rank_events:
        records["notifications"].append(notice)
        records["grants"].append(grant)
    for split, rows in ranks.items():
        records[split].extend(rows)
