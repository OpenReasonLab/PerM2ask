import hashlib
from collections import defaultdict


FACETS = {
    "S1": [
        ("Match the patient identity for this booking.", ["F01"]),
        (
            "Find the registered contact routes used to convey the appointment details.",
            ["F04", "F05"],
        ),
    ],
    "S2": [
        (
            "Review reported symptoms and specified laboratory values for the follow-up.",
            ["F08", "F15"],
        ),
        ("Review the documented recent vital signs for this results visit.", ["F17"]),
    ],
    "S3": [
        (
            "Review drugs in use, recent changes and allergies for reconciliation.",
            ["F11", "F12", "F13"],
        )
    ],
    "S4": [
        (
            "Review reported vaccination history and allergies for screening.",
            ["F11", "F14"],
        ),
        (
            "Review the present symptoms and recent vital signs for screening.",
            ["F08", "F17"],
        ),
    ],
    "S5": [
        ("Review dietary history and documented body measurements.", ["F18", "F19"]),
        (
            "Review the recorded laboratory and vital-sign trajectory pertinent to this follow-up.",
            ["F15", "F17"],
        ),
    ],
    "S6": [
        (
            "Review medications and allergy history for anesthesia preparation.",
            ["F11", "F12", "F13"],
        ),
        (
            "Review the recorded imaging, vital signs and surgical history for this assessment.",
            ["F16", "F17", "F21"],
        ),
    ],
    "S7": [
        (
            "Review the documented functional status and hospitalization course relevant to recovery.",
            ["F20", "F21"],
        ),
        ("Review the recorded vital signs at the postoperative follow-up.", ["F17"]),
    ],
    "S8": [
        ("Review self-reported mental health and sleep history.", ["F22", "F23"]),
        (
            "Review recorded substance use for the agreed mental health follow-up.",
            ["F24"],
        ),
    ],
    "S9": [
        (
            "Review current symptoms and vital signs in this emergency encounter.",
            ["F08", "F17"],
        ),
        (
            "Review allergies, current medicines and recent medication changes.",
            ["F11", "F12", "F13"],
        ),
    ],
}


def stable(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode("utf8")).hexdigest()


def rewrite_memory(record, split):
    subject = record.subject
    docs = {d["document_id"]: d for d in subject["documents"]}
    by_doc = defaultdict(list)
    for fact in subject["facts"]:
        by_doc[fact["document_id"]].append(fact)
    source_lookup = {p["fact_id"]: p for p in record.provenance}
    new_docs = []
    for doc_id, facts in by_doc.items():
        original = docs[doc_id]
        rewritten = {**original, "text": ""}
        ordered = sorted(
            facts, key=lambda f: stable(subject["subject_id"], f["fact_id"], split)
        )
        for fact in ordered:
            src = source_lookup[fact["fact_id"]]

            old = fact["text"]
            attribution = (
                "The intake documents the following observation. "
                if split == "train"
                else "The reviewed chart supplies this entry. "
            )
            statement = attribution + old
            if rewritten["text"]:
                rewritten["text"] += " "
            start = len(rewritten["text"])
            rewritten["text"] += statement
            fact.update(text=statement, start=start, end=len(rewritten["text"]))
            src["transformation"] = f"source_grounded_{split}_vignette"
        new_docs.append(rewritten)
    subject["documents"] = new_docs
    subject["material_kind"] = f"additional_{split}_vignette"


def examples(subject, policy, wordings):
    split = subject["split"]
    uid = subject["subject_id"]
    policy_by_id = {f["id"]: f["labels"] for f in policy["fields"]}
    results = []
    for sidx, scenario in enumerate(policy["scenarios"]):
        sid = scenario["id"]
        authorized = [f for f in subject["facts"] if policy_by_id[f["field_id"]][sidx]]
        variants = [
            w for w in wordings if w["scenario_id"] == sid and w["split"] == split
        ]
        if not variants:
            raise ValueError(f"No checked {split} wording for {sid}.")
        for i, (focus, fields) in enumerate(FACETS[sid]):
            assert all(policy_by_id[field][sidx] for field in fields)
            pos = [f["fact_id"] for f in authorized if f["field_id"] in fields]
            neg = [f["fact_id"] for f in authorized if f["field_id"] not in fields]
            if not pos or not neg:
                continue
            version = variants[int(stable(uid, sid, i), 16) % len(variants)]
            context = f"Scenario: {scenario['name']}. Trusted role: {scenario['role']}. Confirmed task: {version['text']} {focus}"
            results.append(
                {
                    "instance_id": "rank-"
                    + stable(uid, sid, i, version["wording_id"])[:20],
                    "subject_id": uid,
                    "dataset": subject["dataset"],
                    "split": split,
                    "scenario_id": sid,
                    "context": context,
                    "focus": focus,
                    "focus_field_ids": fields,
                    "wording_id": version["wording_id"],
                    "positive_ids": pos,
                    "negative_ids": neg,
                    "supervision": "reconstructed_rule_silver",
                }
            )
    return results
