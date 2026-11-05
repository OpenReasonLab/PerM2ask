import csv
import hashlib
import re
from dataclasses import dataclass

FLAGS = re.I | re.M


EXCLUDED_ROWS = {
    10931,
    11695,
    20730,
    21777,
    24998,
    26160,
    35072,
    40480,
    46406,
    57278,
    59221,
    73226,
    99552,
    101602,
    112077,
    117806,
    123238,
    127572,
}
NONPATIENT = re.compile(
    r"(?:^|\n)\s*(?:breed|species)\s*:|\b(?:veterinary (?:clinic|hospital|medicine)|small animal (?:hospital|clinic)|spayed|neutered|(?:shorthair|persian) cat|(?:crossbreed|shih-tzu) dog|great pyrenees|golden doodle|golden retriever|polar bear|wallaby|body donor|euthanized per protocol)\b",
    FLAGS,
)
MISSING = re.compile(
    r"\b(?:not (?:applicable|disclosed|defined|available|reported|stated|mentioned|provided|specified|indicated|given)|none (?:listed|given|explicitly stated|documented|specified|stated|mentioned|recorded|provided|available|indicated|reported)|unknown|unspecified|confidential|not disclosed|unidentified|to be determined|case report|refer to|medication list available|none applicable|withheld|undisclosed|redacted)\b|^[NX]$|(?:\[|\]|_{2,}|^X{1,4}\b|^.{0,2}\(number\))",
    re.I,
)


ISOLATE = {
    "A 29-year-old man presented to the ED with hemoptysis, central, pleuritic chest pain, and dyspnea": "hemoptysis, central, pleuritic chest pain, and dyspnea",
    "Sudden epigastric pain with vital signs indicating shock": "Sudden epigastric pain",
    "Essential hypertension, high cholesterol, and breast cancer with bilateral mastectomy": "Essential hypertension, high cholesterol, and breast cancer",
    "Hypothyroidism (compensated with daily levothyroxine)": "Hypothyroidism",
    "The patient had Potter syndrome and was on peritoneal dialysis (PD) from 2 weeks of age": "Potter syndrome",
    "Mild mental retardation, diabetes mellitus": "diabetes mellitus",
    "The patient is being treated for systemic hypertension and anemia secondary to renal impairment": "systemic hypertension and anemia secondary to renal impairment",
    "20 pack-year of cigarette smoking, ulcerative colitis": "ulcerative colitis",
    "Asthma (diagnosed 3 years ago), no history of anaphylaxis, urticaria, or drug allergy": "Asthma (diagnosed 3 years ago)",
    "Bronchial asthma, obstructive pulmonary disease, and smoking (15 cigarettes/day)": "Bronchial asthma, obstructive pulmonary disease",
    "Diabetes mellitus, smoking": "Diabetes mellitus",
    "Dyslipidemia, deep vein thrombosis, depression/schizophrenia": "Dyslipidemia, deep vein thrombosis",
    "High blood pressure, hypothyroidism, depression": "High blood pressure, hypothyroidism",
    "Hypertension, no allergies": "Hypertension",
    "Hypertension, previous surgery for breast carcinoma": "Hypertension",
    "Moderate persistent asthma, depression": "Moderate persistent asthma",
    "Obesity, smoking history, chronic alcoholism, and arterial hypertension": "Obesity",
    "Seropositive rheumatoid arthritis (RA), BMI 27.7 kg/m2": "Seropositive rheumatoid arthritis (RA)",
    "Sjögren's syndrome, married, no pregnancy history, no atherosclerosis risk factors": "Sjögren's syndrome",
    "Smoking, high blood pressure, and type 2 diabetes": "high blood pressure, and type 2 diabetes",
    "Diabetes, Right Inguinal Surgery in 2018": "Diabetes",
    "HLH diagnosed in October, 2014, treated with etoposide and dexamethasone": "HLH diagnosed in October, 2014",
    "Fetal alcohol syndrome, congenital ventricular septal defect, chest tube placement": "Fetal alcohol syndrome, congenital ventricular septal defect",
    "Progressive dysphagia, retrosternal pain and weight loss of 10 kg in 2 months": "Progressive dysphagia, retrosternal pain",
    "Atrial fibrillation, shortness of breath, and hypertension": "Atrial fibrillation",
    "Congestive heart failure with left ventricular ejection fraction (LVEF): 40%": "Congestive heart failure",
    "Diabetic for 8 years": "Diabetic for 8 years",
    "Abnormal mammogram, unilateral left arm urticaria": "unilateral left arm urticaria",
    "Fever and Subsegmental Patchy Consolidation with Right Sided Pleural Effusion": "Fever",
    "Menstrual disorders, ovarian tumor": "ovarian tumor",
    "Discharged on low-dose prednisone with no postoperative complications": "no postoperative complications",
    "Discharged with poor general condition and not receiving any additional therapy": "poor general condition",
    "A brain CT showed normal findings, and a transient ischemic attack was diagnosed": "A brain CT showed normal findings",
    "A chest CT revealed a right pneumothorax, and a chest tube was inserted": "A chest CT revealed a right pneumothorax",
    "A chest radiograph showed no tuberculosis, and her laboratory investigations were normal": "A chest radiograph showed no tuberculosis",
    "Abdominal CT demonstrated that the cyst-mass disappeared, and the catheter was removed": "Abdominal CT demonstrated that the cyst-mass disappeared",
    "MRI revealed a temporal tumor, which was excised via left temporal craniotomy": "MRI revealed a temporal tumor",
    "Ultrasound demonstrated a fluid collection and the lesion was percutaneously drained": "Ultrasound demonstrated a fluid collection",
    "Repeat MRI showed complete resolution of the mass with the steroids": "Repeat MRI showed complete resolution of the mass",
    "Ultrasound of the abdomen revealed a left renal mass, which was evaluated further": "Ultrasound of the abdomen revealed a left renal mass",
}
DROP_VALUES = {
    "Ateneol 100mg OD and ACE inhibitor",
    "Not documented",
    "None noted",
    "None received",
    "None relevant",
    "Positive for SARS-CoV-2 antibodies in the patient and his parents",
    "Cervical Orthosis",
    "None required as the patient has passed away",
    "Management of postpartum complications, antiphospholipid syndrome and systemic lupus erythematosus",
    "Alzheimer’s disease, excess weight, and type-2 diabetes",
    "Extensive radius osteomyelitis treated by sequestrectomy, resulting in manus valgus deformity",
    "Patient 1",
    "Patient 2",
    "None (Body Mass Index: 49)",
    "Not Applicable (Body Donor)",
    "Medication list available in the medical record",
    "None prescribed during hospitalization",
    "Inpatient",
    "Outpatient",
    "Ongoing",
    "Admitted with complaints of generalized jaundice and fatigue",
    "Alleged history of consumption of ALP tablets",
    "Ankle foot orthosis for ambulation",
    "Gravida 5, para 5",
    "Keratoconus in older brother",
    "No relevant family history",
    "No known history of drug intake, hormonal changes or mental disorder were reported",
    "No significant contributing family or medical history was reported",
    "No significant medical or family history reported",
    "No specific past dental, family, or medical history elicited",
    "Obesity, obstructive sleep apnea, and non-obstructive coronary artery disease",
    "One previous cesarean section",
    "Previous term delivery and surgically treated ectopic pregnancy",
    "Tract Urinary Infection, Oligohydramnios (during pregnancy)",
    "Ultrasound showed a foetus of 24 weeks period of gestation with polyhydramnios",
    "Pregnancy",
    "Emergency caesarean section for obstructed labour",
    "Positive results in fecal occult blood testing in colon cancer screening",
    "Depression with suicidal ideation",
    "Severe Insomnia and Panic Attacks",
    "Psychotic symptoms",
    "Troublesome voices and self-harm thoughts",
    "Patients presented with nasopharyngeal symptoms after eating raw goat liver",
    "Discharged home",
    "No information provided",
    "Consultation after diagnosis",
    "Genetic evaluation and counseling",
    "Requesting a new set of complete prosthesis",
    "Routine physical examination",
}


RISKY = re.compile(
    r"\b(?:suicid\w*|self-harm|mental health|pregnan\w*|gravida|gestation|alcohol\w*|smok\w*|psychiatr\w*|depressi\w*|schizophren\w*|insomnia|sleep|allerg\w*|family history|father|mother|brother|sister|surgery|medications?|BMI)\b",
    re.I,
)


@dataclass(frozen=True)
class Span:
    field: str
    label: str
    start: int
    end: int
    value: str


def span(note, field, label, start, end):
    while start < end and note[start].isspace():
        start += 1
    while start < end and note[end - 1] in " \t\r\n.;,":
        end -= 1
    return Span(field, label, start, end, note[start:end])


def sanitize(note, field, label, start, end):
    item = span(note, field, label, start, end)
    v = item.value
    token = re.sub(r"[^a-z0-9]+", "", v.casefold())
    if token in {
        "na",
        "nil",
        "unknown",
        "none",
        "x",
        "xx",
        "xxx",
        "notspecified",
        "notrecorded",
        "notreported",
        "notdocumented",
        "unavailable",
        "notavailable",
        "notgiven",
        "notprovided",
        "notsupplied",
        "notapplicable",
        "redacted",
        "notstated",
        "undisclosed",
    }:
        return None, "missing_or_masked_token"
    if field == "F02":
        age = re.match(
            r"\d+(?:\.\d+)?(?:\s*[- ]\s*(?:years?|months?|weeks?|days?|hours?)(?:[- ]old)?(?:\s*(?:,|and)?\s+\d+\s*(?:months?|weeks?|days?))?)?",
            note[item.start :],
            re.I,
        )
        if age:
            item = span(
                note, field, "Age at encounter", item.start, item.start + age.end()
            )
            if (
                re.match(r"\d+\s*(?:years?|[- ]year|\b)", item.value)
                and len(item.value) < 40
            ):
                return item, "age_unit_checked"
        return None, "ambiguous_age"
    if field == "F16" and re.match(r"\.\d", note[item.end :]):
        tail = re.match(r"(?:\.(?=\d)|[^.\n])*", note[item.end :])
        item = span(note, field, label, item.start, item.end + tail.end())
        v = item.value
        if len(v) > 210:
            return None, "truncated_or_compound_measurement"
        if re.search(
            r",?\s+(?:and (?:the patient|she|he|her laboratory)|which (?:was treated|was removed)|thus|followed by|requiring)",
            v,
            re.I,
        ):
            return None, "mixed_imaging_and_course"
        return item, "restore_decimal_and_unit"
    if v in ISOLATE and ISOLATE[v] != v:
        start = item.start + v.index(ISOLATE[v])
        return span(
            note, field, label, start, start + len(ISOLATE[v])
        ), "isolate_single_category"
    if field == "F01" and (
        re.search(
            r"\b(?:doe|unidentified|not|golden|unnamed|unknown|patient|woman|man|baby|XXXX)\b|^(?:ms|mr)\b",
            v,
            re.I,
        )
        or len(v.split()) < 2
    ):
        return None, "placeholder_or_alias_as_name"
    if (
        MISSING.search(v)
        or v in DROP_VALUES
        or v.casefold()
        in (
            "none",
            "nil",
            "n/a",
            "no medication information given",
            "no information available",
        )
        or not v
        or any(x in v for x in "[]{}")
        or "\n" in v
    ):
        return None, "missing_placeholder_or_wrong_category"
    if (
        field == "F15"
        and label.lower() in ("hemoglobin", "haemoglobin")
        and re.search(r"mg/dl", v, re.I)
    ):
        return None, "questionable_source_unit"
    if field == "F08" and RISKY.search(v):
        return None, "mixed_sensitive_presentation"
    if field == "F09" and RISKY.search(v):
        return None, "mixed_or_nonhistorical_detail"
    if field == "F16" and re.search(
        r"gestation|pregnan|surgery|steroid|drainage|catheter was removed|chest tube was inserted",
        v,
        re.I,
    ):
        return None, "mixed_imaging_and_course"
    if (
        field == "F21"
        and label.lower() == "length of stay"
        and not re.fullmatch(
            r"(?:\d+|one) (?:days?|weeks?|hours?|months?|years?)(?: and \d+ months?)?",
            v,
            re.I,
        )
    ):
        return None, "unresolved_duration"
    if field in ("F08", "F09", "F21") and re.search(
        r"\b(?:\d+[- ]year[- ]old|postpartum|antepartum|placenta|labou?r|gestation\w*|cesarean|caesarean|hallucination\w*|delusion\w*|somatoform|anxiety|eating disorder|partial\w* chewed|positive serological testing)\b",
        v,
        re.I,
    ):
        return None, "secondary_protected_category"
    if field == "F12" and re.search(
        r"\b(?:discontinued|passed away|orthosis|brace|physiotherapy)\b", v, re.I
    ):
        return None, "not_an_isolated_current_medication"
    if field == "F12" and len(v) < 2:
        return None, "abbreviated_or_missing_medication"
    if field == "F13" and " were " in v.lower() and " and " not in v.lower():
        context = note[max(0, item.start - 65) : item.start]
        m = re.search(r"\b([A-Za-z][A-Za-z-]{2,22} and )$", context, re.I)
        if m:
            return span(
                note, field, label, item.start - len(m[1]), item.end
            ), "restore_medication_conjunction"
        return None, "truncated_medication_subject"
    if field == "F13":
        prefix = note[max(0, item.start - 40) : item.start]
        qualifier = re.search(
            r"\b(?:low[- ]molecular[- ]weight|unfractionated|intravenous|low[- ]dose|high[- ]dose|oral)\s+$",
            prefix,
            re.I,
        )
        if qualifier:
            return span(
                note, field, label, item.start - len(qualifier[0]), item.end
            ), "restore_drug_formulation"
    return item, "source_scope_checked"


HEADING = {
    "age": ("F02", "Age at encounter"),
    "patient age": ("F02", "Age at encounter"),
    "sex": ("F03", "Recorded sex"),
    "chief complaint": ("F08", "Chief complaint"),
    "presenting complaint": ("F08", "Presenting complaint"),
    "past medical history": ("F09", "Past medical history"),
    "medical history": ("F09", "Medical history"),
    "allergies": ("F11", "Allergies"),
    "family history": ("F10", "Family history"),
    "discharge medications": ("F12", "Discharge medications"),
    "medications at discharge": ("F12", "Discharge medications"),
    "current medications": ("F12", "Current medications"),
    "hemoglobin": ("F15", "Hemoglobin"),
    "haemoglobin": ("F15", "Haemoglobin"),
    "serum sodium": ("F15", "Serum sodium"),
    "hemoglobin a1c": ("F15", "Hemoglobin A1c"),
    "blood pressure": ("F17", "Blood pressure"),
    "heart rate": ("F17", "Heart rate"),
    "temperature": ("F17", "Temperature"),
    "weight": ("F18", "Weight"),
    "height": ("F18", "Height"),
    "bmi": ("F18", "BMI"),
    "diet": ("F19", "Diet"),
    "discharge condition": ("F21", "Discharge condition"),
    "length of stay": ("F21", "Length of stay"),
    "admission diagnosis": ("F21", "Admission diagnosis"),
    "discharge diagnosis": ("F21", "Discharge diagnosis"),
}
HEADING.update(
    {
        k: ("F21", v)
        for k, v in [
            ("admitting diagnosis", "Admission diagnosis"),
            ("primary discharge diagnosis", "Discharge diagnosis"),
            ("principal diagnosis", "Principal diagnosis"),
            ("discharge disposition", "Discharge disposition"),
            ("disposition", "Discharge disposition"),
            ("condition at discharge", "Discharge condition"),
            ("condition on discharge", "Discharge condition"),
            ("hospital outcome", "Hospital outcome"),
            ("reason for admission", "Reason for admission"),
        ]
    }
)
HEADING.update(
    {
        k: ("F15", k.capitalize())
        for k in (
            "serum potassium",
            "sodium",
            "creatinine",
            "platelet count",
            "white blood cell count",
            "serum creatinine",
        )
    }
)
HEADING.update(
    {
        k: ("F17", v)
        for k, v in [
            ("bp", "Blood pressure"),
            ("hr", "Heart rate"),
            ("respiratory rate", "Respiratory rate"),
            ("spo2", "Oxygen saturation"),
        ]
    }
)
HEAD = re.compile(
    r"^[ \t]*[-*]?[ \t]*("
    + "|".join(map(re.escape, HEADING))
    + r")[ \t]*:[ \t]*([^\r\n]+)",
    FLAGS,
)


def additions(note):
    spans = []
    for m in HEAD.finditer(note):
        field, label = HEADING[m[1].lower()]
        value = m[2].strip()
        if (
            len(value) > 100
            or re.search(
                r"(?:because|patient was|years? old|due to|;|<|>|\b(?:unknown|unspecified|unavailable)\b)",
                value,
                re.I,
            )
            and field != "F02"
        ):
            continue
        if field == "F03" and not re.fullmatch(r"male|female", value, re.I):
            continue
        if field in ("F15", "F17", "F18") and not re.fullmatch(
            r"[0-9][\d /.\-\w°²%]{0,20}", value
        ):
            continue
        if field == "F02" and not re.match(r"\d+", value):
            continue
        if (
            field in ("F21", "F08", "F12")
            and RISKY.search(value)
            and label
            not in ("Length of stay", "Current medications", "Discharge medications")
        ):
            continue
        item, _ = sanitize(note, field, label, m.start(2), m.end(2))
        if item:
            spans.append(item)

    for m in re.finditer(
        r"(?im)^(?:patient|age/sex)\s*:\s*(\d{1,3}[- ]year[- ]old)\s+(male|female)\b",
        note[:850],
    ):
        spans.extend(
            (
                span(note, "F02", "Age at encounter", m.start(1), m.end(1)),
                span(note, "F03", "Recorded sex", m.start(2), m.end(2)),
            )
        )
    for m in re.finditer(
        r"(?i)\b(?:the patient(?: is| was|,) a[n]?|patient:\s*)\s*(\d{1,3}[- ](?:year|month|day)s?[- ]old)\s+(male|female|man|woman|boy|girl)\b",
        note[:1250],
    ):
        spans.append(span(note, "F02", "Age at encounter", m.start(1), m.end(1)))
        if m[2].lower() in ("male", "female"):
            spans.append(span(note, "F03", "Recorded sex", m.start(2), m.end(2)))
    return spans


def normal(v):
    return re.sub(r"\W+", " ", v.casefold()).strip()


def templates(item):
    f, l = item.field, item.label.casefold()
    if f == "F21":
        return {
            "length of stay": "How long was the documented hospital stay?",
            "admission diagnosis": "What diagnosis was documented at admission?",
            "discharge disposition": "What discharge disposition is recorded?",
            "principal diagnosis": "What principal diagnosis is documented for this hospitalization?",
            "reason for admission": "What reason for hospital admission is documented?",
            "hospital outcome": "What hospital outcome is recorded?",
            "discharge diagnosis": "What diagnosis was recorded at discharge?",
        }.get(l, "How does the record describe the patient's condition at discharge?")
    if f == "F12":
        return (
            "What does the discharge medication entry list?"
            if "discharge" in l
            else "What is listed in the current medication entry?"
        )
    if f in ("F15", "F17", "F18"):
        return f"What {l} value is documented in this record? Include the unit when stated."
    return {
        "F01": "What patient name appears in this record?",
        "F02": "What age is recorded for the patient at this encounter? Include the unit when stated.",
        "F03": "What sex is recorded for the patient?",
        "F08": "What presenting complaint is documented?",
        "F09": "Which medical-history detail is documented in the profile?",
        "F10": "What family history is reported in this record?",
        "F11": "What is listed in the allergy-history entry?",
        "F13": "What change to medication is documented?",
        "F14": "What vaccination history is documented?",
        "F16": "What does the specified imaging examination report?",
        "F19": "What dietary information is documented?",
        "F20": "What does the record state about functional capacity?",
        "F22": "What mental health information is documented?",
        "F23": "What sleep problem is documented?",
        "F24": "What does the record document about smoking?",
        "F25": "What gestational age is documented?",
    }[f]


def sha(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def choose_spans(originals, alternatives):
    chosen = {i: item for i, item in enumerate(originals) if item is not None}
    fields = {v.field for v in chosen.values()}
    questions = {templates(v) for v in chosen.values()}
    for i, original in enumerate(originals):
        if original is not None:
            continue

        options = alternatives[i]
        for option in options:
            if option.field not in fields and templates(option) not in questions:
                chosen[i] = option
                fields.add(option.field)
                questions.add(templates(option))
                break
        if i not in chosen:
            for option in options:
                if templates(option) not in questions:
                    chosen[i] = option
                    questions.add(templates(option))
                    break
    return chosen


def candidate_fit(fields, notes, chosen_ids):
    best = None
    for key, n in notes.items():
        idx, pid, note, options = n
        if pid in chosen_ids:
            continue

        preferred = [next((s for s in options if s.field == f), None) for f in fields]
        chosen = choose_spans(preferred, [options for _ in fields])
        if len(chosen) != len(fields):
            continue
        score = sum(chosen[i].field == f for i, f in enumerate(fields))
        candidate = (-score, len({templates(v) for v in options}) - len(fields), idx)
        if best is None or candidate < best[0]:
            best = (candidate, n, chosen)
            if score == len(fields) and candidate[1] == 0:
                break
    if best is None:
        raise RuntimeError(
            f"No source-backed human replacement for field set {fields}."
        )
    return best[1:]


def revised_uid(pid):
    import json

    key = json.dumps(("PerM2ask-independent-data-v1", "asclepius", pid), sort_keys=True)
    return "asclepius-" + sha(key)[:16]


def get_sources(raw, source_rows, reserved):
    selected = {}
    alternatives = {}
    unique_notes = set()

    human = re.compile(
        r"\b(?:man|woman|women|girl|boy|infant|child|gentleman|lady|adolescent|newborn|human patient|his wife|her husband|outpatient clinic|emergency department|primary care physician|her physician|his physician)\b",
        re.I,
    )
    with (raw / "asclepius/synthetic.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        for idx, row in enumerate(csv.DictReader(stream)):
            pid, note = row["patient_id"], row["note"]
            if idx in source_rows:
                selected[idx] = (idx, pid, note)
            if (
                len(alternatives) >= 2200
                or pid in reserved
                or pid in alternatives
                or idx in EXCLUDED_ROWS
                or NONPATIENT.search(note)
                or not human.search(note)
            ):
                continue
            opts = additions(note)
            if len({o.field for o in opts}) < 3:
                continue
            signature = sha(normal(note))
            if signature not in unique_notes:
                alternatives[pid] = (idx, pid, note, opts)
                unique_notes.add(signature)
    current = {sha(normal(n[2])) for n in selected.values()}
    return selected, {
        key: n for key, n in alternatives.items() if sha(normal(n[2])) not in current
    }


def aliases(item):
    value = item.value
    result = [value]
    if item.field == "F02":
        readable = re.sub(
            r"(\d+(?:\.\d+)?)\s*[- ](year|month|week|day|hour)s?(?:[- ]old)?",
            lambda m: f"{m[1]} {m[2]}{'s' if m[1] != '1' else ''}",
            value,
            flags=re.I,
        )
        result.append(readable)
    if item.field == "F11" and value.upper() in ("NKDA", "NKA"):
        result.append(
            "no known drug allergies"
            if value.upper() == "NKDA"
            else "no known allergies"
        )
    return list(dict.fromkeys(v for v in result if v))


def revise(data, raw):
    subjects = [s for s in data["subjects"] if s["dataset"] == "asclepius"]
    prov = {
        p["fact_id"]: p
        for p in data["provenance"]
        if p.get("file", "").startswith("asclepius/")
    }
    source_rows = {p["row"] for p in prov.values()}
    reserved = {p["patient_key"] for p in prov.values()}
    selected, pool = get_sources(raw, source_rows, reserved)
    used = set(reserved)
    changed_uid = {}
    link = {}
    for s in subjects:
        old_uid = s["subject_id"]
        facts = s["facts"]
        original_id = [f["fact_id"] for f in facts]
        fields = [f["field_id"] for f in facts]
        pp = [prov[fid] for fid in original_id]
        idx, pid, note = selected[pp[0]["row"]]
        if not all(p["row"] == idx and p["patient_key"] == pid for p in pp):
            raise ValueError(f"Mixed patient provenance for {old_uid}")
        for p in pp:
            assert note[p["source_start"] : p["source_end"]] == p["source_value"]
        force_other = idx in EXCLUDED_ROWS or bool(NONPATIENT.search(note))
        validated = []
        for f, p in zip(facts, pp):
            prefix = re.sub(
                r"^(?:The intake documents the following observation\. |The reviewed chart supplies this entry\. )",
                "",
                f["text"],
            )
            label = prefix.split(":", 1)[0]
            item, _ = sanitize(
                note, f["field_id"], label, p["source_start"], p["source_end"]
            )
            validated.append(item)
        alternatives = additions(note)
        priority = lambda old: (
            [v for v in alternatives if v.field == old]
            + [v for v in alternatives if v.field != old]
        )
        chosen = choose_spans(validated, [priority(field) for field in fields])
        swap = force_other or len(chosen) != len(facts)
        if swap:
            (idx, new_pid, note, _), chosen = candidate_fit(fields, pool, used)
            used.add(new_pid)
            uid = revised_uid(new_pid)
            changed_uid[old_uid] = uid
            s["subject_id"] = uid
            pid = new_pid
        else:
            uid = old_uid
        mapping = {}
        document = s["documents"][0]
        docid = uid + "/profile"
        document.update(document_id=docid, text="")
        for j, (f, p, old_fid) in enumerate(zip(facts, pp, original_id)):
            item = chosen[j]
            fid = old_fid.replace(old_uid, uid)
            mapping[old_fid] = fid
            text = f"{item.label}: {item.value}."
            sep = "\n" if document["text"] else ""
            start = len(document["text"]) + len(sep)
            document["text"] += sep + text
            f.update(
                fact_id=fid,
                document_id=docid,
                field_id=item.field,
                start=start,
                end=start + len(text),
                text=text,
            )
            p.update(
                fact_id=fid,
                subject_id=uid,
                file="asclepius/synthetic.csv",
                row=idx,
                patient_key=pid,
                column="note",
                source_start=item.start,
                source_end=item.end,
                source_value=item.value,
            )
            assert note[item.start : item.end] == item.value
            link[old_fid] = (fid, item, uid)
        for edge in s["source_associations"]:
            edge["members"] = [mapping[k] for k in edge["members"]]
            edge["description_fact_ids"] = [
                mapping[k] for k in edge["description_fact_ids"]
            ]

    for r in data["train"] + data["validation"]:
        if r["dataset"] == "asclepius":
            r["subject_id"] = changed_uid.get(r["subject_id"], r["subject_id"])
            r["positive_ids"] = [link[k][0] for k in r["positive_ids"]]
            r["negative_ids"] = [link[k][0] for k in r["negative_ids"]]
    gold = {g["request_id"]: g for g in data["gold"]}
    policy = {f["id"]: f["labels"] for f in data["policy"]["fields"]}
    seen = set()
    for r in data["requests"]:
        if r["dataset"] != "asclepius":
            continue
        g = gold[r["request_id"]]
        (oldfid,) = g["evidence_ids"]
        fid, item, uid = link[oldfid]
        question = templates(item)
        allowed = policy[item.field][int(r["scenario_id"][1:]) - 1]
        r.update(subject_id=uid, field_id=item.field, question=question)
        g.update(
            allowed=allowed,
            answer=item.value,
            answer_type="text",
            aliases=aliases(item),
            evidence_ids=[fid],
        )
        key = (uid, r["scenario_id"], question)
        if key in seen:
            raise ValueError(f"Ambiguous revised request {r['request_id']} {key}")
        seen.add(key)
    if len(seen) != 10400:
        raise ValueError("Incorrect number of Asclepius test requests")
