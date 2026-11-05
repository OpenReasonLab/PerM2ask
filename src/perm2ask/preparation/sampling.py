import hashlib
from collections import defaultdict


LIST_TASKS = {
    "F09": ("ongoing physical conditions", "ongoing conditions", "conditions"),
    "F10": (
        "conditions in the family medical history",
        "family-history conditions",
        "conditions in the family history",
    ),
    "F11": ("documented allergies", "listed allergies", "allergies"),
    "F12": ("current medications", "listed medications", "medications"),
    "F22": (
        "reported mental health conditions",
        "listed mental health conditions",
        "mental health conditions",
    ),
}
NUMBER_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
]


def digest(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def derived_tasks(record):
    result = []
    for task in record.tasks:
        fid = task["field_id"]
        value = task["answer"]
        variants = []
        if fid in LIST_TASKS:
            label, short, common = LIST_TASKS[fid]
            count = len(value)
            numeral = NUMBER_WORDS[count] if count < len(NUMBER_WORDS) else str(count)
            aliases = [
                f"{n} {noun}" for n in (str(count), numeral) for noun in (short, common)
            ]
            if count == 0:
                aliases.extend((f"no {short}", f"no {common}"))

            variants.append(
                (
                    "count_list",
                    f"How many {label} are listed at intake? Return the number of entries.",
                    count,
                    aliases,
                    {
                        "answer_type": "list",
                        "value": value,
                        "aliases": list(value) if value else [f"no {short}"],
                    },
                )
            )
        if fid == "F20":
            mins = value * 60
            variants.append(
                (
                    "weekly_minutes",
                    "For the activity summary, give the recorded weekly exercise duration in minutes.",
                    mins,
                    [f"{mins} minutes per week", f"{mins} weekly minutes"],
                    {
                        "answer_type": "numeric",
                        "value": value,
                        "aliases": [
                            f"{value} hours per week",
                            f"{value} hours of exercise",
                        ],
                    },
                )
            )
        if fid == "F02":
            for operation, question, answer, phrase in [
                (
                    "under_18",
                    "Does the reported age put this patient in the under-18 group?",
                    value < 18,
                    "the patient is a minor"
                    if value < 18
                    else "the patient is an adult",
                ),
                (
                    "age_65_or_older",
                    "Does the reported age place this patient in the 65-and-older group?",
                    value >= 65,
                    "the patient is at least 65"
                    if value >= 65
                    else "the patient is under 65",
                ),
            ]:
                variants.append(
                    (
                        operation,
                        question,
                        answer,
                        [phrase],
                        {
                            "answer_type": "numeric",
                            "value": value,
                            "aliases": [
                                f"{value} years old",
                                f"age {value}",
                                f"aged {value}",
                            ],
                        },
                    )
                )
        for op, q, answer, aliases, source in variants:
            result.append(
                {
                    "field_id": fid,
                    "question": q,
                    "answer": answer,
                    "aliases": aliases,
                    "evidence_ids": task["evidence_ids"],
                    "derivation": op,
                    "protected_source": source,
                }
            )
    return result


ADDED_FACETS = {
    "S2": [
        (
            "Locate previously reported medical conditions relevant to reviewing the laboratory findings.",
            ["F09"],
        ),
        (
            "Identify the current medicines or recent changes to consider when interpreting the laboratory results.",
            ["F12", "F13"],
        ),
    ],
    "S3": [
        ("Check recorded allergies when reconciling the medication list.", ["F11"]),
        (
            "Reconstruct the active regimen and recent changes for the pharmacist.",
            ["F12", "F13"],
        ),
    ],
    "S4": [
        (
            "Check recorded pregnancy-related information pertinent to this vaccination assessment.",
            ["F25"],
        ),
        (
            "Identify current symptoms reported for the pre-vaccination assessment.",
            ["F08"],
        ),
    ],
    "S5": [
        (
            "Check past medical conditions pertinent to the metabolic follow-up.",
            ["F09"],
        ),
        (
            "Find recorded functional activity to accompany the nutritional follow-up.",
            ["F20"],
        ),
        (
            "Identify ongoing medications documented in the metabolic follow-up record.",
            ["F12"],
        ),
    ],
    "S6": [
        ("Review recorded disease history for the anesthesia assessment.", ["F09"]),
        (
            "Locate the documented surgical and hospitalization course in preparation for anesthesia.",
            ["F21"],
        ),
        (
            "Locate current symptoms recorded for the pre-anesthesia assessment.",
            ["F08"],
        ),
    ],
    "S7": [
        (
            "Identify documented changes to medication during postoperative recovery.",
            ["F13"],
        ),
        (
            "Locate the recorded current symptoms relevant to the postoperative follow-up.",
            ["F08"],
        ),
    ],
    "S8": [
        (
            "Review the currently documented medications during the agreed mental health visit.",
            ["F12"],
        ),
        ("Locate sleep complaints pertinent to the agreed follow-up.", ["F23"]),
    ],
    "S9": [
        (
            "Retrieve the specified imaging and laboratory results for the emergency encounter.",
            ["F15", "F16"],
        ),
        ("Check past disease history to inform the emergency care encounter.", ["F09"]),
        (
            "Locate documented injuries or procedures in the surgical and hospitalization course.",
            ["F21"],
        ),
    ],
}


def more_training(subject, policy, wordings, existing):
    uid = subject["subject_id"]
    candidates = []
    patterns = {(r["scenario_id"], tuple(sorted(r["positive_ids"]))) for r in existing}
    facts = subject["facts"]
    masks = {f["id"]: f["labels"] for f in policy["fields"]}
    for i, s in enumerate(policy["scenarios"]):
        sid = s["id"]
        authorized = [f for f in facts if masks[f["field_id"]][i]]
        wording_options = [
            w for w in wordings if w["scenario_id"] == sid and w["split"] == "train"
        ]
        for focus, fields in ADDED_FACETS.get(sid, []):
            assert all(masks[f][i] for f in fields)
            pos = [f["fact_id"] for f in authorized if f["field_id"] in fields]
            neg = [f["fact_id"] for f in authorized if f["field_id"] not in fields]
            signature = (sid, tuple(sorted(pos)))
            if not pos or not neg or signature in patterns:
                continue
            patterns.add(signature)
            key = digest(uid, sid, focus)
            w = wording_options[int(key, 16) % len(wording_options)]
            candidates.append(
                {
                    "instance_id": "rank-v2-" + key[:20],
                    "subject_id": uid,
                    "dataset": subject["dataset"],
                    "split": "train",
                    "scenario_id": sid,
                    "context": f"Scenario: {s['name']}. Trusted role: {s['role']}. Confirmed task: {w['text']} {focus}",
                    "focus": focus,
                    "focus_field_ids": fields,
                    "wording_id": w["wording_id"],
                    "positive_ids": pos,
                    "negative_ids": neg,
                    "supervision": "reconstructed_rule_silver",
                }
            )
    return sorted(candidates, key=lambda r: digest(r["instance_id"]))


def variable_pools(pools):

    answers = defaultdict(set)
    for pool in pools:
        for request, gold in pool:
            answers[(request["field_id"], gold["derivation"])].add(gold["answer"])
    eligible = {key for key, values in answers.items() if len(values) > 1}
    return [
        [(r, g) for r, g in pool if (r["field_id"], g["derivation"]) in eligible]
        for pool in pools
    ]


def round_robin(pools):
    for i in range(max((len(p) for p in pools), default=0)):
        for pool in pools:
            if i < len(pool):
                yield pool[i]
