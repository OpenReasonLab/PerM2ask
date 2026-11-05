from collections import defaultdict
import hashlib
import json
from pathlib import Path

from .tasks import FACETS
from .sampling import ADDED_FACETS

SPLIT_LEAD = (
    "The intake documents the following observation. ",
    "The reviewed chart supplies this entry. ",
)

NAMES = {
    "F01": "the registered patient name",
    "F02": "the recorded age",
    "F03": "recorded sex",
    "F04": "the listed telephone number",
    "F05": "the listed email address",
    "F06": "the recorded address",
    "F07": "a recorded identifier",
    "F08": "the documented presenting complaint",
    "F09": "the recorded medical history",
    "F10": "the reported family history",
    "F11": "the allergy record",
    "F12": "the medication list",
    "F13": "any documented medication changes",
    "F14": "the immunization record",
    "F15": "the laboratory results",
    "F16": "the recorded imaging findings",
    "F17": "the recorded vital-sign observations",
    "F18": "the recorded body measurements",
    "F19": "the recorded dietary history",
    "F20": "the documented activity or functional-status information",
    "F21": "the available hospitalization or procedure details",
    "F22": "the recorded mental health history",
    "F23": "the documented sleep history",
    "F24": "the documented substance-use history",
    "F25": "the documented reproductive history",
}


def strip_split_headers(subjects):
    changed = set()
    for s in subjects:
        if s["split"] == "test":
            continue
        by_doc = defaultdict(list)
        for f in s["facts"]:
            by_doc[f["document_id"]].append(f)
        for doc in s["documents"]:
            original = doc["text"]
            facts = sorted(by_doc[doc["document_id"]], key=lambda f: f["start"])
            pieces = []
            end = 0
            length = 0
            for f in facts:
                start, stop = f["start"], f["end"]
                if start < end or original[start:stop] != f["text"]:
                    raise ValueError("Invalid non-test source span: " + f["fact_id"])
                gap = original[end:start]
                pieces.append(gap)
                length += len(gap)
                text = f["text"]
                for hdr in SPLIT_LEAD:
                    if text.startswith(hdr):
                        changed.add(f["fact_id"])
                        text = text[len(hdr) :]
                        break
                f.update(start=length, end=length + len(text), text=text)
                pieces.append(text)
                length += len(text)
                end = stop
            doc["text"] = "".join(pieces) + original[end:]
    return changed


def scoped_focus(subject, fields):
    names = [NAMES[f] for f in fields]
    if subject["dataset"] == "ltqa":
        special = {
            "F09": "the self-reported ongoing physical conditions",
            "F12": "items recorded in the intake medication or supplement list",
            "F20": "the self-reported weekly exercise duration",
            "F22": "the self-reported mental health history",
            "F24": "the reported smoking status",
        }
        names = [special.get(f, NAMES[f]) for f in fields]
    listed = (
        names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    )
    return "Find the source entries for " + listed + "."


def revise(data, old_rows, words_path):
    strip_split_headers(data["subjects"])
    subjects = {s["subject_id"]: s for s in data["subjects"]}
    scenarios = {s["id"]: s for s in data["policy"]["scenarios"]}
    policy = {f["id"]: f["labels"] for f in data["policy"]["fields"]}
    words = json.loads(Path(words_path).read_text(encoding="utf8"))
    wordmap = {w["wording_id"]: w for w in words}
    used = set()

    def labels(s, sid, fields):
        column = int(sid[1:]) - 1
        valid = [f for f in s["facts"] if policy[f["field_id"]][column]]
        pos = [f for f in valid if f["field_id"] in fields]
        neg = [f for f in valid if f["field_id"] not in fields]

        positive_text = {f["text"].casefold().strip() for f in pos}
        neg = [f for f in neg if f["text"].casefold().strip() not in positive_text]
        return pos, neg

    def apply(row, s, sid, fields, replaced=False):
        old = old_rows[row["instance_id"]]
        pos, neg = labels(s, sid, fields)
        if not pos or not neg:
            return False
        posids = [f["fact_id"] for f in pos]
        negids = [f["fact_id"] for f in neg]
        sig = (s["subject_id"], sid, frozenset(posids))
        if sig in used:
            return False
        used.add(sig)
        codes = sorted({f["field_id"] for f in pos})
        issues = []
        if set(fields) != set(codes):
            issues.append("FOCUS_LISTS_UNSUPPORTED_FIELDS")
        focus = row["focus"]
        if "trajectory" in focus.lower():
            issues.append("TREND_OR_COMPARISON_UNVERIFIED")
        if (
            "injuries or procedures" in focus.lower()
            or "functional activity" in focus.lower()
        ):
            issues.append("FOCUS_MORE_SPECIFIC_THAN_FIELD")
        if any(
            term in focus.lower()
            for term in (
                "current symptoms",
                "recent vital signs",
                "ongoing medications",
                "active regimen",
            )
        ):
            issues.append("CURRENT_STATUS_NOT_ESTABLISHED")
        if s["dataset"] == "ltqa" and set(codes) & {"F09", "F12", "F20", "F22", "F24"}:
            issues.append("SELF_REPORT_SCOPE")
        if (
            set(old["positive_ids"]) != set(posids)
            or set(old["negative_ids"]) != set(negids)
            or old["subject_id"] != s["subject_id"]
        ):
            issues.append("RECHECKED_SOURCE_LABELS")
        if replaced:
            issues.append("RETARGETED_INVALID_OR_DUPLICATE_PAIR")
        if issues or replaced:
            focus = scoped_focus(s, codes)
        chosen = wordmap.get(row["wording_id"])
        if not chosen or chosen["split"] != s["split"] or chosen["scenario_id"] != sid:
            variants = [
                w for w in words if w["split"] == s["split"] and w["scenario_id"] == sid
            ]
            chosen = variants[
                int(hashlib.sha256(row["instance_id"].encode()).hexdigest(), 16)
                % len(variants)
            ]
        sc = scenarios[sid]
        ctx = f"Scenario: {sc['name']}. Trusted role: {sc['role']}. Confirmed task: {chosen['text']} {focus}"
        row.update(
            subject_id=s["subject_id"],
            dataset=s["dataset"],
            split=s["split"],
            scenario_id=sid,
            context=ctx,
            focus=focus,
            focus_field_ids=codes,
            wording_id=chosen["wording_id"],
            positive_ids=posids,
            negative_ids=negids,
            supervision="reconstructed_rule_silver",
        )
        return True

    pending = []
    for split in ("train", "validation"):
        for r in data[split]:
            s = subjects.get(r["subject_id"])
            if (
                not s
                or s["split"] != split
                or not apply(r, s, r["scenario_id"], r["focus_field_ids"])
            ):
                pending.append((split, r))
    for split, r in pending:
        pool = sorted(
            (
                s
                for s in subjects.values()
                if s["dataset"] == r["dataset"] and s["split"] == split
            ),
            key=lambda s: (s["subject_id"] != r["subject_id"], s["subject_id"]),
        )
        found = False
        for s in pool:
            for sid in sorted(scenarios, key=lambda sid: sid != r["scenario_id"]):
                for _, fields in list(FACETS.get(sid, [])) + list(
                    ADDED_FACETS.get(sid, [])
                ):
                    if apply(r, s, sid, fields, True):
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            raise ValueError(f"No source-grounded replacement for {r['instance_id']}")
