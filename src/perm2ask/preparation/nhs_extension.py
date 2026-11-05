import re
from collections import defaultdict
from itertools import combinations

from .nhs import DATASET, REG, define_task, identify, negative_status, plain_time
from .tasks import stable


def reviewed_items(subject, sources):
    items = []
    for fact in subject["facts"]:
        fid = fact["fact_id"]
        source = sources[fid]
        raw_value = source["source_value"]
        value = source.get("reviewed_value", raw_value)
        time = source.get("documentation_time")
        title = None
        if source["file"] == REG:
            kind = {"F01": "name", "F02": "age"}[fact["field_id"]]
            label = "registered " + kind
        else:
            header = re.fullmatch(
                r"The (.+?) note entered on (.+?) lists ([^:]+): (.+)\.",
                fact["text"],
                re.DOTALL,
            )
            if not header or header[2] != plain_time(time):
                raise ValueError(f"Invalid reviewed NHS note header: {fid}")
            title, label = header[1], header[3]
            if fact["field_id"] == "F17":
                kind = "vital"
            else:
                kind = "negative" if negative_status(value) else "clinical"
        items.append(
            {
                "fact_id": fid,
                "subject_id": subject["subject_id"],
                "field_id": fact["field_id"],
                "title": title,
                "time": time,
                "note_id": source.get("source_document_id"),
                "label": label,
                "kind": kind,
                "raw_value": raw_value,
                "value": value,
                "unit": source.get("explicit_unit", ""),
                "issues": [],
            }
        )
    return items


def balanced(pools, count):
    buckets = [
        sorted(rows, key=lambda row: row[0], reverse=True)
        for _, rows in sorted(pools.items())
    ]
    result = []
    while buckets and len(result) < count:
        for bucket in buckets:
            result.append(bucket.pop()[1])
            if len(result) == count:
                return result
        buckets = [bucket for bucket in buckets if bucket]
    raise ValueError(f"Only {len(result)} unique NHS candidates for {count} rows")


def paired_entries(items, ambiguities, scenarios, permissions):
    by_note = defaultdict(list)
    for item in items:
        if item["note_id"]:
            by_note[item["note_id"]].append(item)
    pairs = []
    seen = set()
    for note_id, entries in sorted(by_note.items()):
        for left, right in combinations(
            sorted(entries, key=lambda i: i["field_id"]), 2
        ):
            if left["field_id"] == right["field_id"]:
                continue
            signature = tuple(
                (i["title"], i["time"], i["label"], i["value"]) for i in (left, right)
            )
            if signature in seen:
                continue
            seen.add(signature)
            fields, ids, components = [], [], []
            for item in (left, right):
                _, value, aliases, target = define_task([item], ambiguities)
                fields.append(item["field_id"])
                ids.append(item["fact_id"])
                components.append(
                    {
                        "key": item["label"].replace(" ", "_"),
                        "field_id": item["field_id"],
                        "evidence_ids": [item["fact_id"]],
                        "answer": value,
                        "aliases": aliases,
                        "nhs_target": target,
                    }
                )
            question = (
                f"From the {left['title']} note entered on {plain_time(left['time'])} "
                f"(note ID {note_id}), return the transcribed {left['label']} and {right['label']} "
                f"as an object with keys {components[0]['key']} and {components[1]['key']}. "
                "Use the wording in the supplied record."
            )
            if right["kind"] == "vital":
                question += " Give the vital reading without units."
            for scenario in scenarios:
                sid = scenario["id"]
                column = int(sid[1:]) - 1
                allowed = permissions[fields[0]][column]
                if allowed != permissions[fields[1]][column]:
                    continue
                rid = "req-" + stable("nhs-note-pair", sid, *ids)[:20]
                request = {
                    "request_id": rid,
                    "subject_id": left["subject_id"],
                    "dataset": DATASET,
                    "split": "test",
                    "scenario_id": sid,
                    "field_id": fields[0],
                    "field_ids": fields,
                    "question": question,
                }
                gold = {
                    "request_id": rid,
                    "allowed": allowed,
                    "answer": {c["key"]: c["answer"] for c in components},
                    "answer_type": "object",
                    "aliases": [],
                    "evidence_ids": ids,
                    "derivation": "paired_note_entries",
                    "nhs_target": {"kind": "note_pair", "components": components},
                }
                pairs.append((request, gold))
    return pairs


def new_requests(data, subjects, reviewed, permissions):
    keys = {
        item["fact_id"]: (
            subject_id,
            item["time"],
            item["title"],
            item["label"],
            item["value"],
        )
        for subject_id, items in reviewed.items()
        for item in items
    }
    existing_gold = {row["request_id"]: row for row in data["gold"]}
    used_questions = set()
    used_evidence = set()
    original_single = {}
    for row in data["requests"]:
        if row["dataset"] != DATASET:
            continue
        gold = existing_gold[row["request_id"]]
        ids = tuple(gold["evidence_ids"])
        used_questions.add((row["subject_id"], row["scenario_id"], row["question"]))
        used_evidence.add((row["scenario_id"], tuple(keys[fid] for fid in ids)))
        if len(ids) == 1:
            original_single[ids[0]] = row["question"]

    pools = defaultdict(list)
    clinical = []
    for sub in subjects:
        if sub["split"] != "test":
            continue
        subject_id = sub["subject_id"]
        items = reviewed[subject_id]
        ambiguities = defaultdict(set)
        for item in items:
            ambiguities[subject_id, item["time"], item["title"], item["label"]].add(
                item["value"]
            )
        clinical.extend(
            paired_entries(items, ambiguities, data["policy"]["scenarios"], permissions)
        )
        representatives = {}
        for item in sorted(items, key=lambda item: item["fact_id"]):
            representatives.setdefault(keys[item["fact_id"]], item)
        eligible = list(representatives.values())
        candidates = [(item,) for item in eligible]
        vital_groups = defaultdict(list)
        for item in eligible:
            if item["kind"] == "vital":
                vital_groups[item["label"]].append(item)
        for group in vital_groups.values():
            for first, second in combinations(
                sorted(group, key=lambda item: item["time"]), 2
            ):
                if (
                    first["time"] != second["time"]
                    and first["note_id"] != second["note_id"]
                    and first["value"] != second["value"]
                ):
                    candidates.append((first, second))
        for members in candidates:
            first = members[0]
            field = first["field_id"]
            ids = tuple(item["fact_id"] for item in members)
            question, answer, aliases, target = define_task(members, ambiguities)
            if len(ids) == 1:
                if ids[0] in original_single:
                    question = original_single[ids[0]]
                elif first["kind"] in ("negative", "clinical"):
                    question = (
                        f"What {first['label']} entry is transcribed from "
                        f"{identify(first, ambiguities)}? Use the wording in the supplied record."
                    )
            for scenario in data["policy"]["scenarios"]:
                sid = scenario["id"]
                evidence_key = (sid, tuple(keys[fid] for fid in ids))
                if (
                    evidence_key in used_evidence
                    or (subject_id, sid, question) in used_questions
                ):
                    continue
                used_evidence.add(evidence_key)
                used_questions.add((subject_id, sid, question))
                digest = stable("nhs-extension", sid, *ids)
                rid = "req-" + digest[:20]
                request = {
                    "request_id": rid,
                    "subject_id": subject_id,
                    "dataset": DATASET,
                    "split": "test",
                    "scenario_id": sid,
                    "field_id": field,
                    "question": question,
                }
                gold = {
                    "request_id": rid,
                    "allowed": permissions[field][int(sid[1:]) - 1],
                    "answer": answer,
                    "answer_type": "text",
                    "aliases": aliases,
                    "evidence_ids": list(ids),
                    "nhs_target": target,
                }
                if first["kind"] == "vital":
                    pools[subject_id, sid, len(ids)].append((digest, (request, gold)))
                else:
                    clinical.append((request, gold))
    clinical.sort(key=lambda pair: pair[0]["request_id"])
    return clinical + balanced(pools, 3323 - len(clinical))


def new_rankings(rows, subjects, reviewed, permissions, count):
    templates = {}
    seen = set()
    for row in rows:
        if row["dataset"] == DATASET:
            templates.setdefault(row["scenario_id"], row)
            seen.add(
                (row["subject_id"], row["scenario_id"], frozenset(row["positive_ids"]))
            )
    pools = defaultdict(list)
    for sub in subjects:
        subject_id = sub["subject_id"]
        text = {f["fact_id"]: f["text"].casefold().strip() for f in sub["facts"]}
        for sid, template in sorted(templates.items()):
            valid = [
                item
                for item in reviewed[subject_id]
                if permissions[item["field_id"]][int(sid[1:]) - 1]
            ]
            groups = defaultdict(list)
            for item in valid:
                groups[item["label"], None].append(item)
                if item["time"]:
                    groups[item["label"], item["time"]].append(item)
            for (label, time), members in groups.items():
                positive = sorted(item["fact_id"] for item in members)
                sig = (subject_id, sid, frozenset(positive))
                if sig in seen:
                    continue
                seen.add(sig)
                positive_text = {text[fid] for fid in positive}
                negative = sorted(
                    item["fact_id"]
                    for item in valid
                    if text[item["fact_id"]] not in positive_text
                )
                if not negative:
                    continue
                focus = f"Find the source entries for {label}"
                if time:
                    focus += f" in notes entered on {plain_time(time)}"
                focus += "."
                prefix = template["context"].removesuffix(template["focus"])
                digest = stable("nhs-ranking-extension", subject_id, sid, *positive)
                row = dict(
                    template,
                    instance_id="rank-" + digest[:20],
                    subject_id=subject_id,
                    focus=focus,
                    context=prefix + focus,
                    positive_ids=positive,
                    negative_ids=negative,
                    focus_field_ids=[members[0]["field_id"]],
                )
                pools[subject_id, sid].append((digest, row))
    return balanced(pools, count)


def extend(data):
    for key, expected in (("requests", 3323), ("train", 131), ("validation", 44)):
        if sum(row["dataset"] == DATASET for row in data[key]) != expected:
            raise ValueError(f"NHS extension requires {expected} original {key} rows")
    subjects = [s for s in data["subjects"] if s["dataset"] == DATASET]
    sources = {
        row["fact_id"]: row
        for row in data["provenance"]
        if row["subject_id"].startswith(DATASET + "-")
    }
    reviewed = {s["subject_id"]: reviewed_items(s, sources) for s in subjects}
    permissions = {field["id"]: field["labels"] for field in data["policy"]["fields"]}
    pairs = new_requests(data, subjects, reviewed, permissions)
    rankings = {}
    for split, count in (("train", 131), ("validation", 44)):
        rankings[split] = new_rankings(
            data[split],
            [s for s in subjects if s["split"] == split],
            reviewed,
            permissions,
            count,
        )
    data["requests"].extend(row for row, _ in pairs)
    data["gold"].extend(gold for _, gold in pairs)
    for split, rows in rankings.items():
        data[split].extend(rows)
