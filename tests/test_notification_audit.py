import importlib.util
import sys
from pathlib import Path

import pytest

from perm2ask.notifications import fixture_channels

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "notification_audit", ROOT / "scripts/verify_notifications.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def case(profile):
    scenario = {
        "id": "S6",
        "purpose": "Complete the assessment.",
        "role": "anesthesia clinician",
    }
    row = {
        "request_id": "notify-1",
        "subject_id": "patient-1",
        "scenario_id": "S6",
        "pair_id": "pair-1",
        "notification_id": "event-1",
    }
    fields = {"F02"}
    notice, grant = fixture_channels(
        row, scenario, {"F09"} if profile == "narrow_grant" else fields, profile
    )
    citation = {
        "notification_id": "event-1",
        "pair_id": "pair-1",
        "grant_id": grant["grant_id"],
        "profile": profile,
    }
    return row, citation, notice, grant, scenario, fields, {"F02", "F09"}


@pytest.mark.parametrize("profile", sorted(audit.NEGATIVES | {"valid"}))
def test_signed_profiles_are_checked_independently(profile):
    args = case(profile)
    actor, active, scope = audit.check_channel_pair(*args)
    assert actor == (
        profile not in {"wrong_actor", "bad_signature", "expired_notification"}
    )
    assert active == (profile in {"valid", "narrow_grant"})
    assert ("F02" in scope) == (profile == "valid")
    assert ("F09" in scope) == (profile == "narrow_grant")


@pytest.mark.parametrize(
    "key", ["request_id", "subject_id", "pair_id", "notification_id"]
)
def test_verifier_rejects_replayed_event(key):
    args = case("valid")
    args[0][key] += "-other"
    with pytest.raises(ValueError):
        audit.check_channel_pair(*args)


def test_expected_failure_cannot_disguise_an_extra_failure():
    args = case("bad_signature")
    args[1]["profile"] = "revoked_grant"
    with pytest.raises(ValueError):
        audit.check_channel_pair(*args)


def test_replacing_v4_prefix_with_semantically_equivalent_json_fails():
    original = b'{"x":1}\r\n'
    pin = {"bytes": len(original), "sha256": audit.sha(original), "prefix_rows": 1}
    audit.verify_pin(original + b'{"x":2}\n', pin, "requests")
    with pytest.raises(ValueError):
        audit.verify_pin(b'{"x": 1}\n', pin, "requests")


def test_source_citations_check_the_selected_profile_member():
    gold = {"request_id": "notify-q", "evidence_ids": ["f1"]}
    source = {
        "f1": {
            "file": audit.RAW_FILE,
            "row": 0,
            "column": "profile.age",
            "source_value": 20,
            "patient_key": "synthetic A",
        }
    }
    pointer = {"fact_id": "f1", "row": 0, "column": "profile.age"}
    citation = {
        "request_id": "notify-q",
        "source_file": audit.RAW_FILE,
        "source_revision": audit.RAW_REVISION,
        "source_sha256": audit.RAW_SHA,
        "evidence_ids": ["f1"],
        "evidence": [pointer],
    }
    raw = [
        {
            "field": "name",
            "ref_answer": "synthetic A",
            "profile": {"name": "synthetic A", "age": 20},
        }
    ]
    audit.check_citation(citation, gold, source, raw)
    pointer["column"] = "profile.name"
    with pytest.raises(ValueError):
        audit.check_citation(citation, gold, source, raw)
