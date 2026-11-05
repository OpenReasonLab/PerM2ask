import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "asclepius": (10400, 696, 113),
    "nhs_synthetic": (6646, 262, 88),
    "ltqa": (4586, 104, 40),
}
SPLITS = ("test", "train", "validation")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def index(rows, key):
    indexed = {row[key]: row for row in rows}
    require(len(rows) == len(indexed), f"Duplicate {key}")
    return indexed


def verify_files(data):
    pins = read_json(ROOT / "configs/base_sha256.json")
    current = read_json(data / "audit.json")["sha256"]
    require(pins.keys() <= current.keys(), "Release audit file coverage")
    for name, pin in pins.items():
        payload = (data / name).read_bytes()
        require(
            hashlib.sha256(payload).hexdigest() == current[name],
            f"Release hash: {name}",
        )
        if "prefix_rows" in pin:
            payload = payload[: pin["bytes"]]
            require(payload.count(b"\n") == pin["prefix_rows"], name)
        require(
            hashlib.sha256(payload).hexdigest() == pin["sha256"], f"v3 mismatch: {name}"
        )


def check_raw(raw, provenance):
    for name, manifest in read_json(ROOT / "data/sources.json").items():
        with (raw / name).open("rb") as stream:
            digest = hashlib.sha256()
            while block := stream.read(1 << 20):
                digest.update(block)
        require(digest.hexdigest() == manifest["sha256"], name)
    by_file = defaultdict(lambda: defaultdict(list))
    for entry in provenance.values():
        by_file[entry["file"]][entry["row"]].append(entry)
    checked = 0
    for name, needed in by_file.items():
        with (raw / name).open(encoding="utf-8-sig", newline="") as stream:
            rows = (
                json.load(stream) if name.endswith(".json") else csv.DictReader(stream)
            )
            for position, row in enumerate(rows):
                for entry in needed.get(position, ()):
                    fid = entry["fact_id"]
                    if name.startswith("ltqa/"):
                        profile = row["profile"]
                        subject_key = profile["name"]
                        value = profile[entry["column"].removeprefix("profile.")]
                    else:
                        subject_key = row.get("person_id", row.get("patient_id"))
                        value = row[entry["column"]]
                    require(
                        subject_key == entry["patient_key"], f"Source subject: {fid}"
                    )
                    if "source_start" in entry:
                        start, end = entry["source_start"], entry["source_end"]
                        require(0 <= start < end <= len(value), f"Raw span: {fid}")
                        if "clinical_excerpt" in entry:
                            require(
                                row["clinical_note_id"] == entry["source_document_id"],
                                fid,
                            )
                            require(row["admission_id"] == entry["encounter_id"], fid)
                            a, b = (
                                entry["clinical_excerpt_start"],
                                entry["clinical_excerpt_end"],
                            )
                            require(
                                value[a:b] == entry["clinical_excerpt"],
                                f"Excerpt: {fid}",
                            )
                        value = value[start:end]
                    require(value == entry["source_value"], f"Source value: {fid}")
                    checked += 1
    require(checked == len(provenance), "Missing raw source rows")
    return checked


def check_subjects(subjects, provenance, splits):
    require(set(subjects) == set(splits), "Subject split coverage")
    facts = {}
    identities = {}
    for uid, subject in subjects.items():
        require(splits[uid] == subject["split"] in SPLITS, uid)
        docs = index(subject["documents"], "document_id")
        local = index(subject["facts"], "fact_id")
        for fid, fact in local.items():
            require(fid not in facts and provenance[fid]["subject_id"] == uid, fid)
            facts[fid] = fact
            source = provenance[fid]
            identity = (subject["dataset"], source["patient_key"])
            require(
                identities.setdefault(identity, uid) == uid, f"Shared patient: {uid}"
            )
            document = docs[fact["document_id"]]["text"]
            a, b = fact["start"], fact["end"]
            require(0 <= a < b <= len(document) and document[a:b] == fact["text"], fid)
    require(facts.keys() == provenance.keys(), "Fact/provenance coverage")
    return facts


def owned_evidence(row, ids, provenance):
    require(ids and len(ids) == len(set(ids)), f"Invalid evidence: {row}")
    require(
        all(provenance[i]["subject_id"] == row["subject_id"] for i in ids),
        "Evidence ownership",
    )


def check_new_pair(row, subject, facts, policy, provenance):
    sid = row["scenario_id"]
    authorized = [f for f in subject["facts"] if policy[f["field_id"]][sid]]
    if row["dataset"] == "nhs_synthetic":
        focus = re.fullmatch(
            r"Find the source entries for (.+?)(?: in notes entered on (.+?))?\.",
            row["focus"],
        )
        require(focus, row["instance_id"])
        label, time = focus.groups()
        positives = set()
        for f in authorized:
            source = provenance[f["fact_id"]]
            if "documentation_time" in source:
                reference, tail = f["text"].split(" lists ", 1)
                match = tail.startswith(label + ": ") and (
                    not time or reference.endswith(time)
                )
            else:
                match = (
                    not time
                    and label
                    == {"F01": "registered name", "F02": "registered age"}[
                        f["field_id"]
                    ]
                )
            if match:
                positives.add(f["fact_id"])
    else:
        positives = {
            f["fact_id"] for f in authorized if f["field_id"] in row["focus_field_ids"]
        }
    require(positives == set(row["positive_ids"]), f"Rank focus: {row['instance_id']}")
    texts = {facts[i]["text"].strip().casefold() for i in positives}
    negatives = {
        f["fact_id"] for f in authorized if f["text"].strip().casefold() not in texts
    }
    require(
        negatives == set(row["negative_ids"]), f"Rank negatives: {row['instance_id']}"
    )


def check_rank(rows, subjects, provenance, facts, policy, split, wordings):
    seen = set()
    ids = set()
    counts = Counter()
    for position, row in enumerate(rows):
        uid, sid = row["subject_id"], row["scenario_id"]
        rid = row["instance_id"]
        require(rid not in ids, rid)
        ids.add(rid)
        require(subjects[uid]["split"] == row["split"] == split, rid)
        require(subjects[uid]["dataset"] == row["dataset"], rid)
        counts[row["dataset"]] += 1
        if position >= {"train": 879, "validation": 177}[split]:
            check_new_pair(row, subjects[uid], facts, policy, provenance)
        positive, negative = row["positive_ids"], row["negative_ids"]
        require(positive and negative, rid)
        owned_evidence(row, positive + negative, provenance)
        require(
            all(policy[facts[i]["field_id"]][sid] for i in positive + negative), rid
        )
        key = (uid, sid, frozenset(positive))
        require(key not in seen, f"Duplicate rank target: {rid}")
        seen.add(key)
        fields = {facts[i]["field_id"] for i in positive}
        require(fields == set(row["focus_field_ids"]), rid)
        positive_text = {facts[i]["text"].strip().casefold() for i in positive}
        require(
            all(
                facts[i]["text"].strip().casefold() not in positive_text
                for i in negative
            ),
            rid,
        )
        wording = wordings[row["wording_id"]]
        require(wording["split"] == split and wording["scenario_id"] == sid, rid)
        require(
            row["focus"] in row["context"] and wording["text"] in row["context"], rid
        )
    return counts, ids


LTQA_KEYS = {
    "F01": "name",
    "F02": "age",
    "F04": "phone",
    "F05": "email",
    "F06": "address",
    "F07": "ssn",
    "F09": "physical_conditions",
    "F10": "family_history",
    "F11": "allergies",
    "F12": "medicines_or_supplements",
    "F19": "diet",
    "F20": "exercise_hours",
    "F22": "mental_health",
    "F24": "current_smoking",
}
FOOD = {
    "wheat",
    "gluten",
    "nuts",
    "pine nuts",
    "peanuts",
    "shellfish",
    "kiwifruit",
    "soy",
}


def source_value(entry):
    return entry.get("reviewed_value", entry["source_value"])


def same(value, expected, rid):
    require(
        json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True),
        f"Wrong source-derived value: {rid}",
    )


def check_ltqa(request, gold, facts, sources):
    rid = request["request_id"]
    refs = gold["evidence_ids"]
    op = gold.get("derivation")
    values = {facts[i]["field_id"]: source_value(sources[i]) for i in refs}
    if op in {"paired_intake", "intake_abstract"}:
        components = gold["components"]
        require(2 <= len(refs) == len(values) == len(components) <= 3, rid)
        expected = {}
        require(request["field_ids"] == [facts[fid]["field_id"] for fid in refs], rid)
        for component, fid in zip(components, refs):
            field = facts[fid]["field_id"]
            key = LTQA_KEYS[field]
            require(component["field_id"] == field and component["key"] == key, rid)
            require(
                component["evidence_ids"] == [fid] and key in request["question"], rid
            )
            same(component["answer"], values[field], rid)
            expected[key] = values[field]
        same(gold["answer"], expected, rid)
        require(gold["answer_type"] == "object", rid)
        semantic_op = "return_fields"
    else:
        require(len(refs) == 1, rid)
        value = next(iter(values.values()))
        if op == "weekly_minutes":
            value = value * 60
        elif op == "count_list":
            value = len(value)
        elif op == "has_listed_entry":
            value = bool(value)
        elif op == "food_allergen_listed":
            value = bool(FOOD & {item.casefold() for item in value})
        else:
            require(op is None, f"Unknown intake operation: {op}")
        same(gold["answer"], value, rid)
        semantic_op = op or "return_fields"
    if op:
        require(len(gold["protected_sources"]) == len(refs), rid)
        for part, fid in zip(gold["protected_sources"], refs):
            field = facts[fid]["field_id"]
            require(part["field_id"] == field and part["evidence_ids"] == [fid], rid)
            same(part["value"], values[field], rid)
    return semantic_op, tuple(sorted(values))


def check_nhs(request, gold, facts, sources, new):
    rid = request["request_id"]
    refs = gold["evidence_ids"]
    target = gold["nhs_target"]
    entries = [sources[i] for i in refs]
    values = [source_value(entry) for entry in entries]
    same(target["readings"], values, rid)
    same(
        target["source_units"],
        [entry.get("explicit_unit", "") for entry in entries],
        rid,
    )
    same(gold["answer"], " then ".join(values), rid)
    normalized = []
    for fid, entry, value in zip(refs, entries, values):
        label = target["label"]
        if "documentation_time" in entry:
            note_ref, tail = facts[fid]["text"].split(" lists ", 1)
            require(tail.startswith(label + ": "), rid)
            require(value in tail, rid)
            if new:
                require(note_ref[4:].casefold() in request["question"].casefold(), rid)
                if target["kind"] == "vital":
                    require(label in request["question"].casefold(), rid)
            normalized.append((note_ref, label, value))
        else:
            require(value in facts[fid]["text"], rid)
            normalized.append((facts[fid]["field_id"], "", value))
    if new and len(refs) > 1:
        require(len(refs) == 2 and target["kind"] == "vital", rid)
        require(
            entries[0]["source_document_id"] != entries[1]["source_document_id"], rid
        )
        times = [entry["documentation_time"].replace("T", " at ") for entry in entries]
        require(
            request["question"].index(times[0]) < request["question"].index(times[1]),
            rid,
        )
        require(
            values[0] != values[1]
            and entries[0]["documentation_time"] < entries[1]["documentation_time"],
            rid,
        )
    return "return_readings", tuple(sorted(normalized))


def check_note_pair(request, gold, facts, sources):
    rid = request["request_id"]
    refs = gold["evidence_ids"]
    parts = gold["nhs_target"]["components"]
    require(
        gold["nhs_target"]["kind"] == "note_pair" and gold["answer_type"] == "object",
        rid,
    )
    require(len(refs) == len(parts) == len(set(request["field_ids"])) == 2, rid)
    require(len({sources[i]["source_document_id"] for i in refs}) == 1, rid)
    expected = {}
    signature = []
    for fid, field, part in zip(refs, request["field_ids"], parts):
        require(part["field_id"] == field == facts[fid]["field_id"], rid)
        require(part["evidence_ids"] == [fid], rid)
        key = part["nhs_target"]["label"].replace(" ", "_")
        require(part["key"] == key and key in request["question"], rid)
        expected[key] = source_value(sources[fid])
        _, single = check_nhs(request, part, facts, sources, True)
        signature.extend(single)
    same(gold["answer"], expected, rid)
    return "return_note_fields", tuple(sorted(signature))


def check_requests(data, subjects, facts, sources, policy):
    legacy_count = sum(target[0] for target in TARGETS.values())
    requests = read_rows(data / "benchmark/requests.jsonl")[:legacy_count]
    indexed = index(requests, "request_id")
    gold = index(read_rows(data / "private/gold.jsonl")[:legacy_count], "request_id")
    require(indexed.keys() == gold.keys(), "Gold/request ID coverage")
    seen, prompts = set(), set()
    counts = Counter()
    added = Counter()
    variation = defaultdict(set)
    old_intake_groups = set()
    for position, request in enumerate(requests):
        rid, uid = request["request_id"], request["subject_id"]
        dataset, sid = request["dataset"], request["scenario_id"]
        require(subjects[uid]["dataset"] == dataset, rid)
        require(request["split"] == subjects[uid]["split"] == "test", rid)
        g = gold[rid]
        refs = g["evidence_ids"]
        owned_evidence(request, refs, sources)
        fields = {facts[i]["field_id"] for i in refs}
        require(fields == set(request.get("field_ids", [request["field_id"]])), rid)
        require({policy[f][sid] for f in fields} == {g["allowed"]}, rid)
        counts[dataset] += 1
        new = position >= 16016
        if dataset == "nhs_synthetic":
            if g.get("derivation") == "paired_note_entries":
                signature = check_note_pair(request, g, facts, sources)
                op = g["derivation"]
            else:
                signature = check_nhs(request, g, facts, sources, new)
                op = "reading_pair" if len(refs) == 2 else g["nhs_target"]["kind"]
        elif dataset == "ltqa":
            signature = check_ltqa(request, g, facts, sources)
            op = g.get("derivation", "direct")
            group = (sid, tuple(sorted(fields)))
            if not new and len(fields) > 1:
                old_intake_groups.add(group)
            if new:
                require(op == "intake_abstract" and group not in old_intake_groups, rid)
                variation[group].add(json.dumps(g["answer"], sort_keys=True))
            if op not in {"direct", "paired_intake", "intake_abstract"}:
                variation[(op, request["field_id"])].add(json.dumps(g["answer"]))
        else:
            require(not new, rid)
            continue
        signature = (uid, sid, signature)
        prompt = (uid, sid, " ".join(request["question"].casefold().split()))
        if new:
            require(
                signature not in seen and prompt not in prompts, f"Repeated task: {rid}"
            )
            added[(dataset, op)] += 1
        seen.add(signature)
        prompts.add(prompt)
    require(
        all(len(values) > 1 for values in variation.values()), "Constant intake family"
    )
    return (
        counts,
        {f"{d}/{op}": n for (d, op), n in sorted(added.items())},
        set(indexed),
    )


def verify(data, raw):
    verify_files(data)
    subjects = index(read_rows(data / "benchmark/subjects.jsonl"), "subject_id")
    provenance = index(read_rows(data / "private/provenance.jsonl"), "fact_id")
    facts = check_subjects(subjects, provenance, read_json(data / "splits.json"))
    raw_count = check_raw(raw, provenance) if raw is not None else None
    policy_data = read_json(data / "policy.json")
    scenario_ids = [scenario["id"] for scenario in policy_data["scenarios"]]
    policy = {
        f["id"]: dict(zip(scenario_ids, f["labels"])) for f in policy_data["fields"]
    }
    wordings = index(read_json(ROOT / "configs/scenario_wordings.json"), "wording_id")
    counts, operations, ids = check_requests(data, subjects, facts, provenance, policy)
    per_split = {"test": counts}
    for i, split in enumerate(SPLITS[1:], 1):
        rows = read_rows(data / f"private/ranker_{split}.jsonl")
        rows = rows[: sum(target[i] for target in TARGETS.values())]
        per_split[split], split_ids = check_rank(
            rows, subjects, provenance, facts, policy, split, wordings
        )
        require(not ids.intersection(split_ids), "Repeated ID between partitions")
        ids.update(split_ids)
    for i, split in enumerate(SPLITS):
        expected = {d: targets[i] for d, targets in TARGETS.items()}
        require(
            per_split[split] == expected,
            f"{split}: expected {expected}, observed {per_split[split]}",
        )
    return {
        "verified": True,
        "subjects": len(subjects),
        "facts": len(facts),
        "raw_facts_verified": raw_count,
        "counts": per_split,
        "new_test_operations": operations,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Independently verify the expanded release against v3."
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--raw", type=Path, help="Directory containing the eight pinned upstream files"
    )
    args = parser.parse_args()
    result = {"scenario_only": verify(args.data, args.raw)}
    if "trusted/notifications.jsonl" in read_json(args.data / "audit.json")["sha256"]:
        from verify_notifications import verify as verify_notifications

        result["notifications"] = verify_notifications(args.data, args.raw)
    print(json.dumps(result, indent=2))
