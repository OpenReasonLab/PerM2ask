from collections import defaultdict
from datetime import datetime
from pathlib import Path
import csv
import re

DATASET = "nhs_synthetic"
VITALS = {
    "blood pressure": (r"(?:BP|blood pressure)", r"\d{2,3}\s*/\s*\d{2,3}", r"mm\s?Hg"),
    "heart rate": (r"(?:HR|heart rate)", r"\d{2,3}(?:\.\d+)?", r"(?:bpm|beats/min)"),
    "respiratory rate": (
        r"(?:RR|respiratory rate)",
        r"\d{1,2}(?:\.\d+)?",
        r"(?:br/min|breaths/min|/min|bpm)",
    ),
    "temperature": (
        r"(?:temp|temperature)",
        r"\d{2,3}(?:\.\d+)?",
        r"(?:Â?°\s*)?[CF]\b",
    ),
    "oxygen saturation": (
        r"(?:SpO2|Sp02|oxygen saturation)",
        r"\d{2,3}(?:\.\d+)?",
        r"%",
    ),
}
TEXT_FIELDS = {
    "chief complaint": "F08",
    "presenting complaint": "F08",
    "past medical history": "F09",
    "family history": "F10",
    "allergies": "F11",
    "current medications": "F12",
    "functional status": "F20",
    "sleep status": "F23",
}
NOTES = "nhs_synthetic/synthetic_clinical_notes.csv"
REG = "nhs_synthetic/patients.csv"


def read_csv(raw, relative):
    with (Path(raw) / relative).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def timestamp(raw_time):
    return datetime.strptime(raw_time, "%d/%m/%Y %H:%M").isoformat(timespec="minutes")


def plain_time(iso):
    return iso.replace("T", " at ")


def negative_status(value):
    return value.strip().casefold() in {
        "none",
        "none reported",
        "none declared",
        "nka",
        "nkda",
        "nil",
        "no current medications",
    }


def clinical_atoms(value):
    if negative_status(value):
        return []
    atoms = []
    for part in re.split(r"[,;]|\s+and\s+", value, flags=re.I):
        part = part.strip(" .")
        if len(part) > 2:
            atoms.append(part)

    vocabulary = (
        "pollen",
        "nuts",
        "penicillin",
        "aspirin",
        "sulfonamides",
        "macrolides",
        "paracetamol",
        "ibuprofen",
        "atorvastatin",
        "bisoprolol",
        "ferrous sulfate",
        "omeprazole",
        "osteoarthritis",
        "asthma",
        "hyperlipidemia",
        "osteoporosis",
        "HTN",
        "HLD",
        "seasonal allergies",
        "allergic rhinitis",
        "DDH",
        "breathlessness",
        "poor feeding",
        "headache",
        "chest pain",
        "abdominal pain",
        "wheeze",
        "nausea",
        "photophobia",
        "confusion",
    )
    for term in vocabulary:
        if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", value, re.I):
            atoms.append(term)
    return list(dict.fromkeys(atoms))


def read_header(text, start):
    line_start = text.rfind("\n", 0, start) + 1
    prefix = text[line_start:start]
    return re.sub(r"^[\s*\-]+", "", prefix).split(":")[0].strip().casefold()


def vital_components(text, start, measure):
    _, number, unit = VITALS[measure]
    match = re.match(number, text[start:])
    if not match:
        raise ValueError("invalid raw NHS measurement at mapped offset")
    value = match[0]
    end = start + len(value)
    suffix = text[end:].split("\n", 1)[0].rstrip("\r")
    unit_match = re.match(r"\s*(" + unit + r")", suffix, re.I)
    raw_unit = unit_match[1] if unit_match else ""
    clean_unit = raw_unit.replace("Â°", "°")
    display = value + (" " + clean_unit if clean_unit else "")
    end_excerpt = end + (unit_match.end() if unit_match else 0)
    qualification = ""
    if measure == "oxygen saturation":
        raw_suffix = text[end_excerpt:].split("\n", 1)[0].strip(" \t.\r")
        if raw_suffix:
            qualification = raw_suffix
            end_excerpt = text.find(qualification, end_excerpt) + len(qualification)
            normalized = {
                "on roomair": "on room air",
                "on rooom air": "on room air",
                "on roo m air": "on room air",
                "onair": "on air",
            }.get(qualification.casefold(), qualification)
            if qualification.casefold() in {"on da", "(ar)", "on zair", "pon air"}:
                normalized = (
                    f'(delivery qualifier transcribed as "{qualification}"; unresolved)'
                )
            display += " " + normalized
    return value, start, end, display, clean_unit, qualification, end_excerpt


def clinical_components(text, start, end, label, field):
    value = text[start:end]
    issues = []

    if field == "F12" and re.search(r"\s*\(initiated\b", value, re.I):
        value = re.split(r"\s*\(initiated\b", value, 1, flags=re.I)[0].rstrip()
        issues.append("medication_change_separated_from_current_medication")
    if field == "F12" and re.match(
        r"paracetamol\s+500mg\s+PRN\s+for pain,", value, re.I
    ):
        value = re.split(r"\s+for pain,", value, 1, flags=re.I)[0].rstrip()
        issues.append("symptom_indication_separated_from_medication_field")
    if field == "F11" and ". Confirmed and documented" in value:
        value = value.split(". Confirmed and documented", 1)[0]
        issues.append("editorial_confirmation_removed_from_allergen_list")

    if field == "F09" and re.match(
        r"asthma,\s*(?:seasonal allergies|seasonal allergic rhinitis)", value, re.I
    ):
        value = value.split(",", 1)[0]
        issues.append("allergy_item_separated_from_history_field")
    return value, start, start + len(value), issues


def review_fact(
    fact, doc, source, notes, registrations, admissions, duplicates, registered_names
):
    original = source["source_value"]
    issues = []
    if source["file"] == REG:
        row = registrations[source["row"]]
        assert (
            row["person_id"] == source["patient_key"]
            and row[source["column"]] == original
        )
        assert (source["column"], fact["field_id"]) in {
            ("age", "F02"),
            ("full_name", "F01"),
        }
        label = "registered age" if fact["field_id"] == "F02" else "registered name"
        return dict(
            value=original,
            label=label,
            kind="age" if fact["field_id"] == "F02" else "name",
            unit="",
            note_id=None,
            time=None,
            title=None,
            duplicate_count=1,
            issues=issues,
        )
    assert source["file"] == NOTES
    row = notes[source["row"]]
    text = row["clean_note_text"]
    start, end = source["source_start"], source["source_end"]
    assert text[start:end] == original and row["person_id"] == source["patient_key"]
    assert admissions[row["admission_id"]]["patient_id"] == row["person_id"]
    time = timestamp(row["creation_timestamp"])
    assert (row["admission_id"], row["clinical_note_id"], time) == (
        source["encounter_id"],
        source["source_document_id"],
        source["timestamp"],
    )
    assert (
        doc["document_id"].endswith("/" + row["clinical_note_id"])
        and doc["encounter_id"] == row["admission_id"]
        and doc["timestamp"] == time
    )
    named = re.search(r"(?mi)^\s*(?:-\s*)?patient name\s*:\s*([^\r\n]+)", text)
    if named:
        printed = named[1].split(",")[0].strip(" .").casefold()
        if printed not in registered_names[row["person_id"]]:
            source["note_name_disagrees_with_registration"] = True
            issues.append(
                "note_name_spelling_differs_verified_by_patient_and_admission_ids"
            )
    header = read_header(text, start)
    unit, qualification = "", ""
    if fact["field_id"] == "F17":
        label = next(
            k for k, (h, _, _) in VITALS.items() if re.fullmatch(h, header, re.I)
        )
        value, start, end, display, unit, qualification, end_excerpt = vital_components(
            text, start, label
        )
        if not original.startswith(value):
            issues.append("truncated_measurement_recovered")
        if unit and unit not in original:
            issues.append("explicit_source_unit_recovered")
        if qualification:
            issues.append("oxygen_delivery_qualifier_restored")
        if qualification.casefold() in {"on da", "(ar)", "on zair", "pon air"}:
            issues.append("unclear_oxygen_delivery_spelling_left_uninterpreted")

        kind = "vital"
    else:
        label = header
        assert TEXT_FIELDS[label] == fact["field_id"]
        value, start, end, notes_issues = clinical_components(
            text, start, end, label, fact["field_id"]
        )
        issues.extend(notes_issues)
        display = re.sub(r"\bcataarct\b", "cataract", value, flags=re.I)
        display = re.sub(r"\bweks\b", "weeks", display, flags=re.I)
        if display != value:
            source["reviewed_value"] = display
            issues.append("unambiguous_source_typo_normalized")
        end_excerpt = end
        kind = "negative" if negative_status(value) else "clinical"
    copies = duplicates[(row["person_id"], time, text)]
    if len(copies) > 1:
        issues.append("duplicate_note_same_documentation_time")
    issues.append("documentation_time_not_measurement_time")
    new = f"The {row['note_subject']} note entered on {plain_time(time)} lists {label}: {display}."
    fact["text"] = new
    source.update(
        source_start=start,
        source_end=end,
        source_value=value,
        documentation_time=time,
        clinical_excerpt_start=start,
        clinical_excerpt_end=end_excerpt,
        clinical_excerpt=text[start:end_excerpt],
        explicit_unit=unit,
        same_time_copy_count=len(copies),
    )
    assert (
        text[start:end] == value
        and text[start:end_excerpt] == source["clinical_excerpt"]
    )
    return dict(
        value=source.get("reviewed_value", value),
        raw_value=value,
        label=label,
        kind=kind,
        note_id=row["clinical_note_id"],
        time=time,
        title=row["note_subject"],
        encounter=row["admission_id"],
        duplicate_count=len(copies),
        raw_text=text,
        unit=unit,
        qualifier=qualification,
        issues=issues,
    )


def relink_document(doc, changes):
    old = doc["text"]
    chunks = []
    offset = 0
    position = 0
    for start, end, f, old_text in sorted(changes, key=lambda v: v[0]):
        assert old[start:end] == old_text and start >= position
        gap = old[position:start]
        chunks.append(gap)
        offset += len(gap)
        f["start"] = offset
        chunks.append(f["text"])
        offset += len(f["text"])
        f["end"] = offset
        position = end
    chunks.append(old[position:])
    doc["text"] = "".join(chunks)
    for _, _, f, _ in changes:
        assert doc["text"][f["start"] : f["end"]] == f["text"]


def identify(item, ambiguities):
    name = f"the {item['title']} note entered on {plain_time(item['time'])}"

    if (
        len(ambiguities[item["subject_id"], item["time"], item["title"], item["label"]])
        > 1
    ):
        name += f" (note ID {item['note_id']})"
    return name


def define_task(items, ambiguities):
    first = items[0]
    kind, label = first["kind"], first["label"]
    answers = [v["value"] for v in items]
    if kind == "name":
        question = "What full name appears in the patient registration record?"
    elif kind == "age":
        question = "What numeric age appears in the patient registration record?"
    elif len(items) == 2:
        refs = [identify(v, ambiguities) for v in items]
        reading = (
            "systolic/diastolic pair"
            if label == "blood pressure"
            else "numerical reading"
        )
        question = (
            f"Compare the {label} entries in {refs[0]} and {refs[1]}. "
            f"Give each {reading} in that order, without units, even if the values agree."
        )
    elif kind == "vital":
        reading = "systolic/diastolic pair" if label == "blood pressure" else "number"
        question = f"What {label} {('pair' if label == 'blood pressure' else 'value')} appears in {identify(first, ambiguities)}? Give the {reading} without units."
    elif "allergy_item_separated_from_history_field" in first["issues"]:
        question = f"What condition appears first under past medical history in {identify(first, ambiguities)}?"
    elif "symptom_indication_separated_from_medication_field" in first["issues"]:
        question = f"What first medication, dose and dosing instruction appear under current medications in {identify(first, ambiguities)}? Omit the indication."
    elif label == "current medications":
        question = f"What is listed under current medications in {identify(first, ambiguities)}? Give the current entry and omit any treatment-change history."
    elif label == "allergies":
        question = f"What allergy or allergy status is listed in {identify(first, ambiguities)}? Use the documented wording."
    else:
        question = f"What does {identify(first, ambiguities)} list under {label}? Use the documented wording."
    meta = {
        "kind": kind,
        "label": label,
        "readings": answers,
        "source_units": [v["unit"] for v in items],
    }
    atoms = list(dict.fromkeys(answers))
    if kind == "clinical":
        atoms += clinical_atoms(answers[0])
        if first["raw_value"] != first["value"]:
            meta["equivalent_texts"] = [first["raw_value"]]
            atoms.append(first["raw_value"])
    elif kind == "vital":
        atoms.extend(f"{v['value']} {v['unit']}" for v in items if v["unit"])
    return question, " then ".join(answers), list(dict.fromkeys(atoms)), meta


def revise(data, raw):
    notes = read_csv(raw, NOTES)
    registrations = read_csv(raw, REG)
    admissions = {
        r["admission_id"]: r for r in read_csv(raw, "nhs_synthetic/admissions.csv")
    }
    registered_names = defaultdict(set)
    for person in registrations:
        registered_names[person["person_id"]].add(
            person["full_name"].strip(" .").casefold()
        )
    duplicates = defaultdict(list)
    for n in notes:
        duplicates[
            n["person_id"], timestamp(n["creation_timestamp"]), n["clean_note_text"]
        ].append(n["clinical_note_id"])
    sources = {v["fact_id"]: v for v in data["provenance"]}
    fact_info = {}
    ambiguities = defaultdict(set)
    for sub in data["subjects"]:
        if sub["dataset"] != DATASET:
            continue
        docs = {d["document_id"]: d for d in sub["documents"]}
        changes = defaultdict(list)
        for f in sub["facts"]:
            original = (f["start"], f["end"], f, f["text"])
            src = sources[f["fact_id"]]
            assert src["subject_id"] == sub["subject_id"]
            result = review_fact(
                f,
                docs[f["document_id"]],
                src,
                notes,
                registrations,
                admissions,
                duplicates,
                registered_names,
            )
            result["subject_id"] = sub["subject_id"]
            result["field_id"] = f["field_id"]
            fact_info[f["fact_id"]] = result
            ambiguities[
                sub["subject_id"], result["time"], result["title"], result["label"]
            ].add(result["value"])
            changes[f["document_id"]].append(original)
        for did, items in changes.items():
            relink_document(docs[did], items)
        for edge in sub["source_associations"]:
            source_items = [fact_info[fid] for fid in edge["members"]]
            if all(i["kind"] == "vital" for i in source_items):
                single_doc = len({i["note_id"] for i in source_items}) == 1
                edge["description"] = (
                    "Vital signs listed in one clinical note."
                    if single_doc
                    else f"{source_items[0]['label'].capitalize()} entries in notes ordered by documentation time."
                )
    gold = {g["request_id"]: g for g in data["gold"]}
    labels = {f["id"]: f["labels"] for f in data["policy"]["fields"]}
    keys = set()
    for q in data["requests"]:
        if q["dataset"] != DATASET:
            continue
        rid = q["request_id"]
        g = gold[rid]
        items = [fact_info[f] for f in g["evidence_ids"]]
        assert 1 <= len(items) <= 2 and all(
            i["subject_id"] == q["subject_id"] for i in items
        )
        assert g["allowed"] == labels[q["field_id"]][int(q["scenario_id"][1:]) - 1]
        assert all(i["field_id"] == q["field_id"] for i in items)
        if len(items) == 2:
            assert (
                items[0]["label"] == items[1]["label"]
                and items[0]["kind"] == items[1]["kind"] == "vital"
            )
            assert (
                items[0]["time"] < items[1]["time"]
                and items[0]["note_id"] != items[1]["note_id"]
            )
            assert items[0]["raw_text"] != items[1]["raw_text"]
        new_q, answer, aliases, meta = define_task(items, ambiguities)
        assert (q["subject_id"], q["scenario_id"], new_q) not in keys
        keys.add((q["subject_id"], q["scenario_id"], new_q))
        g.update(answer=answer, aliases=aliases, nhs_target=meta)
        q["question"] = new_q
    if len(keys) != 3323:
        raise ValueError("Incorrect number of NHS Synthetic test requests")
