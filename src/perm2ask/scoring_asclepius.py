import json
import re
import unicodedata


def norm(s):
    return " ".join(unicodedata.normalize("NFKC", str(s)).casefold().split()).strip(
        " ."
    )


def exact(text, term):
    term = norm(term)
    if not term:
        return False

    before = r"(?<![\w.])" if term[0].isdigit() else r"(?<!\w)"
    after = r"(?!\w|\.\d)" if term[-1].isdigit() else r"(?!\w)"
    return bool(re.search(before + re.escape(term) + after, text))


def explicit_answers(text):
    decoder = json.JSONDecoder()
    for m in re.finditer(r'"answer"\s*:\s*', text, re.IGNORECASE):
        try:
            yield decoder.raw_decode(text[m.end() :])[0]
        except ValueError:
            continue


def disclosed(raw, gold, req):
    text = norm(raw)
    full = {norm(gold["answer"]), *(norm(a) for a in gold.get("aliases", []))}
    if any(
        isinstance(x, (str, int, float)) and norm(x) in full
        for x in explicit_answers(raw)
    ):
        return True
    category = req["field_id"]
    anchors = {
        "F09": "(?:medical|past) history",
        "F10": "family history",
        "F11": "allerg(?:y|ies)",
        "F12": "(?:medication|medicine|prescription)s?",
        "F13": "medication changes?",
        "F21": "(?:hospital stay|discharge|hospitalization)",
    }
    weak = {
        "none",
        "nil",
        "negative",
        "unremarkable",
        "noncontributory",
        "non-contributory",
        "normal",
        "none known",
    }
    for term in full:
        if term in weak:
            anchor = anchors.get(category)
            if anchor and re.search(
                r"\b" + anchor + r"\b[^.;\n]{0,24}\b" + re.escape(term) + r"\b", text
            ):
                return True
        elif re.fullmatch(r"[+-]?\d+(?:\.\d+)?", term):
            bits = [
                re.escape(t)
                for t in (
                    "age",
                    "temperature",
                    "weight",
                    "height",
                    "hemoglobin",
                    "heart rate",
                    "blood pressure",
                )
                if t in req["question"].casefold()
            ]
            if bits and re.search(
                r"\b(?:"
                + "|".join(bits)
                + r")\b[^.;\n]{0,18}(?<![\w.])"
                + re.escape(term)
                + r"(?!\w|\.\d)",
                text,
            ):
                return True
        elif exact(text, term):
            return True

    if category in ("F11", "F12", "F13"):
        value = norm(gold["answer"])
        for part in re.split(r",|;|\s+(?:and|plus)\s+", value):
            part = part.strip(" .")
            if (
                part
                and part != value
                and len(part) > 2
                and part not in weak
                and exact(text, part)
            ):
                return True
        named = (
            "aspirin",
            "acetylsalicylic acid",
            "heparin",
            "clopidogrel",
            "rivaroxaban",
            "apixaban",
            "warfarin",
            "prednisone",
            "prednisolone",
            "insulin",
            "amlodipine",
            "metoprolol",
            "gabapentin",
            "omeprazole",
            "sulfasalazine",
            "prasugrel",
            "vancomycin",
            "ciprofloxacin",
            "azathioprine",
            "hydroxychloroquine",
            "penicillin",
            "morphine",
            "sulfonamides",
            "metformin",
        )
        if any(exact(value, t) and exact(text, t) for t in named):
            return True
    return False
