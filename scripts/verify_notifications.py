import argparse
import base64
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from verify_expansion import (
    ROOT,
    SPLITS,
    check_ltqa,
    check_subjects,
    index,
    owned_evidence,
    read_json,
    read_rows,
    require,
    source_value,
)

TARGETS = {
    "test": {"asclepius": 10400, "nhs_synthetic": 6646, "ltqa": 9172},
    "train": {"asclepius": 696, "nhs_synthetic": 262, "ltqa": 208},
    "validation": {"asclepius": 113, "nhs_synthetic": 88, "ltqa": 80},
}
BASE_ROWS = {"test": 21632, "train": 1062, "validation": 241}
RAW_FILE = "ltqa/airgapagent-r.json"
RAW_REVISION = "f1d8f27376eb0854ec5b476e88b942d0060eee62"
RAW_SHA = "117a76be7caa1e0cde682a0f594fb70af8c6e6859041bc8a5b39c48b22bdf3e3"
EVALUATION_TIME = 1793836800
AUDIENCE = "perm2ask-simulated-runtime"


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def verify_pin(content, pin, name):
    if "prefix_rows" in pin:
        content = content[: pin["bytes"]]
        require(content.count(b"\n") == pin["prefix_rows"], f"Prefix rows: {name}")
    require(
        len(content) == pin["bytes"] and sha(content) == pin["sha256"],
        f"v4 bytes: {name}",
    )


def verify_release_files(data):
    pins = read_json(ROOT / "configs/v4_base_sha256.json")
    manifest = read_json(data / "audit.json")["sha256"]
    require(pins.keys() | SIDECARS == manifest.keys(), "Incorrect audit file coverage")
    for name, checksum in manifest.items():
        payload = (data / name).read_bytes()
        require(sha(payload) == checksum, f"Release checksum: {name}")
        if name in pins:
            verify_pin(payload, pins[name], name)


def raw_profiles(raw):
    manifest = read_json(ROOT / "data/sources.json")[RAW_FILE]
    require(
        manifest["revision"] == RAW_REVISION and manifest["sha256"] == RAW_SHA, RAW_FILE
    )
    if raw is None:
        return None
    content = (raw / RAW_FILE).read_bytes()
    require(sha(content) == RAW_SHA, "AirGapAgent-R upstream digest")
    return json.loads(content)


def check_citation(citation, gold, provenance, raw):
    rid = gold["request_id"]
    require(citation["request_id"] == rid and citation["source_file"] == RAW_FILE, rid)
    require(
        citation["source_revision"] == RAW_REVISION
        and citation["source_sha256"] == RAW_SHA,
        rid,
    )
    expected_ids = gold["evidence_ids"]
    require(citation["evidence_ids"] == expected_ids, rid)
    require([v["fact_id"] for v in citation["evidence"]] == expected_ids, rid)
    for pointer in citation["evidence"]:
        fid = pointer["fact_id"]
        source = provenance[fid]
        require(source["file"] == RAW_FILE and pointer["row"] == source["row"], rid)
        require(
            pointer["column"] == source["column"]
            and source["column"].startswith("profile."),
            rid,
        )
        if raw is not None:
            profile = raw[source["row"]]["profile"]
            require(profile["name"] == source["patient_key"], rid)
            upstream = profile[pointer["column"].removeprefix("profile.")]
            require(upstream == source["source_value"], rid)
            if "reviewed_value" in source:
                require(source["column"] == "profile.current_medications", rid)
                require(
                    source_value(source)
                    == [v for v in upstream if v != "CPAP for Sleep Apnea"],
                    rid,
                )


def decode_segment(segment):
    require(
        isinstance(segment, str) and segment and "=" not in segment,
        "Malformed JWS segment",
    )
    value = base64.b64decode(
        segment + "=" * (-len(segment) % 4), altchars=b"-_", validate=True
    )
    require(
        base64.urlsafe_b64encode(value).rstrip(b"=").decode() == segment,
        "Noncanonical JWS encoding",
    )
    return value


def read_channel(token, issuer, key):
    header, body, signature = token.split(".")
    require(
        json.loads(decode_segment(header))
        == {"alg": "EdDSA", "typ": "JWT", "kid": issuer},
        "JWS header",
    )
    claims = json.loads(decode_segment(body))
    valid_signature = True
    try:
        Ed25519PublicKey.from_public_bytes(key).verify(
            decode_segment(signature), (header + "." + body).encode("ascii")
        )
    except InvalidSignature:
        valid_signature = False
    return claims, valid_signature


def check_counts(rows, split):
    observed = Counter(row["dataset"] for row in rows)
    require(observed == TARGETS[split], f"Unexpected {split} counts: {observed}")
    require(all(row["split"] == split for row in rows), f"Foreign {split} row")
    return observed


def check_anchor(request, gold, anchor, original, subjects, facts, provenance, policy):
    rid = request["request_id"]
    require(request["dataset"] == "ltqa" and request["split"] == "test", rid)
    require(subjects[request["subject_id"]]["split"] == "test", rid)
    for key in (
        "dataset",
        "split",
        "subject_id",
        "scenario_id",
        "field_id",
        "field_ids",
        "question",
    ):
        require(request.get(key) == anchor.get(key), f"Anchor changed {key}: {rid}")
    require(original["allowed"] is True, rid)
    require(set(original) == set(gold) and type(gold["allowed"]) is bool, rid)
    for key, value in original.items():
        if key not in ("request_id", "allowed"):
            require(
                json.dumps(gold[key], sort_keys=True)
                == json.dumps(value, sort_keys=True),
                f"Target {key}: {rid}",
            )
    owned_evidence(request, gold["evidence_ids"], provenance)
    fields = {facts[i]["field_id"] for i in gold["evidence_ids"]}
    require(fields == set(request.get("field_ids", [request["field_id"]])), rid)
    require(
        all(policy[f][request["scenario_id"]] for f in fields),
        f"Anchor exceeds policy: {rid}",
    )
    check_ltqa(request, gold, facts, provenance)
    return fields


PUBLIC_KEYS = {
    "fixture-notifier": bytes.fromhex(
        "df7f7a3565edf7ee1f95ec901643f36572d52fb1b7777987d9ab9f2c228cc7eb"
    ),
    "fixture-grants": bytes.fromhex(
        "40bc77b80c88ae6852ae83ca2e28cc5ccf117d9a3a66aa0af1e4ba8850f161cb"
    ),
}
NEGATIVES = {
    "wrong_actor",
    "bad_signature",
    "expired_notification",
    "revoked_grant",
    "narrow_grant",
    "wrong_grant_subject",
}


def within_window(claims):
    nbf, exp = claims["nbf"], claims["exp"]
    require(
        type(nbf) is int and type(exp) is int and exp > nbf, "Malformed fixture window"
    )
    return nbf <= EVALUATION_TIME < exp


def check_bound_claims(claims, subject, scenario, issuer):
    require(
        claims["iss"] == issuer and claims["aud"] == AUDIENCE,
        "Fixture issuer or audience",
    )
    require(
        claims["scenario"] == scenario["id"]
        and claims["role"] == scenario["role"]
        and claims["purpose"] == scenario["purpose"],
        "Fixture scenario binding",
    )
    return claims["sub"] == subject, claims["actor"] == "fixture-app-" + scenario["id"]


def check_channel_pair(
    request, citation, notification, grant, scenario, fields, ceiling
):
    rid = request.get("request_id") or request["instance_id"]
    profile = citation["profile"]
    require(profile in NEGATIVES | {"valid"}, f"Unknown fixture profile: {profile}")
    issuer, owner = "fixture-notifier", "fixture-grants"
    event, signed = read_channel(notification["token"], issuer, PUBLIC_KEYS[issuer])
    consent, grant_signed = read_channel(grant["token"], owner, PUBLIC_KEYS[owner])
    if "instance_id" in request:
        require(event["focus"] == request["focus"], rid)
    common = {
        "iss",
        "aud",
        "nbf",
        "exp",
        "sub",
        "role",
        "purpose",
        "actor",
        "rid",
        "scenario",
        "grant_id",
    }
    expected_event = common | {"notification_id", "pair_id"}
    if "instance_id" in request:
        expected_event.add("focus")
    require(
        set(event) == expected_event
        and set(consent) == common | {"status", "allowed_fields"},
        rid,
    )
    require(event["nbf"] == consent["nbf"] == EVALUATION_TIME - 3600, rid)
    require(consent["exp"] == EVALUATION_TIME + 3600, rid)
    require(
        event["exp"]
        == (
            EVALUATION_TIME
            if profile == "expired_notification"
            else EVALUATION_TIME + 3600
        ),
        rid,
    )
    event_subject, event_actor = check_bound_claims(
        event, request["subject_id"], scenario, issuer
    )
    grant_subject, grant_actor = check_bound_claims(
        consent, request["subject_id"], scenario, owner
    )
    require(
        event_subject and grant_actor and grant_signed and within_window(consent), rid
    )
    require(event["rid"] == consent["rid"] == rid, f"Event replay: {rid}")
    require(
        event["notification_id"]
        == request["notification_id"]
        == citation["notification_id"]
        == notification["notification_id"],
        rid,
    )
    require(event["pair_id"] == request["pair_id"] == citation["pair_id"], rid)
    require(
        event["grant_id"]
        == citation["grant_id"]
        == consent["grant_id"]
        == grant["grant_id"],
        rid,
    )
    require(signed == (profile != "bad_signature"), f"Notification signature: {rid}")
    require(event_actor == (profile != "wrong_actor"), rid)
    require(grant_subject == (profile != "wrong_grant_subject"), rid)
    require(within_window(event) == (profile != "expired_notification"), rid)
    require(
        consent["status"] == ("revoked" if profile == "revoked_grant" else "active"),
        rid,
    )
    grant_fields = consent["allowed_fields"]
    require(
        isinstance(grant_fields, list) and len(grant_fields) == len(set(grant_fields)),
        rid,
    )
    require(set(grant_fields) <= ceiling, rid)
    if profile != "narrow_grant":
        require(set(grant_fields) == fields, rid)
    permitted = ceiling & set(grant_fields)
    require(
        fields <= permitted if profile != "narrow_grant" else not (fields & permitted),
        rid,
    )
    event_ok = event_subject and event_actor and within_window(event) and signed
    grant_ok = (
        grant_subject
        and grant_actor
        and within_window(consent)
        and grant_signed
        and consent["status"] == "active"
    )
    return (
        event_ok,
        grant_ok and event_ok,
        permitted if event_ok and grant_ok else set(),
    )


def check_ranker(rows, split, subjects, provenance, facts, policy):
    check_counts(rows, split)
    signatures = set()
    cohort = set()
    for position, row in enumerate(rows):
        rid, uid, sid = row["instance_id"], row["subject_id"], row["scenario_id"]
        require(
            subjects[uid]["split"] == split
            and subjects[uid]["dataset"] == row["dataset"],
            rid,
        )
        positive, negative = row["positive_ids"], row["negative_ids"]
        require(positive and negative, rid)
        owned_evidence(row, positive + negative, provenance)
        require(
            all(policy[facts[fid]["field_id"]][sid] for fid in positive + negative), rid
        )
        signature = (uid, sid, frozenset(positive))
        require(signature not in signatures, f"Repeated ranking positives: {rid}")
        signatures.add(signature)
        if position < BASE_ROWS[split]:
            continue
        require(row["dataset"] == "ltqa", rid)
        authorized = [f for f in subjects[uid]["facts"] if policy[f["field_id"]][sid]]
        focus_fields = set(row["focus_field_ids"])
        expected = {f["fact_id"] for f in authorized if f["field_id"] in focus_fields}
        require(set(positive) == expected, rid)
        require(
            set(negative) <= {f["fact_id"] for f in authorized} - expected
            and len(negative) <= 2,
            rid,
        )
        require(
            set(row["focus_field_ids"]) == {facts[i]["field_id"] for i in positive}, rid
        )
        require(row["focus"] in row["context"], rid)
        cohort.add(uid)
    return cohort


DIMENSIONS = {
    "wrong_actor": "notification_actor",
    "bad_signature": "notification_signature",
    "expired_notification": "notification_validity",
    "revoked_grant": "grant_status",
    "narrow_grant": "grant_scope",
    "wrong_grant_subject": "grant_subject",
}
SIDECARS = {
    "trusted/notifications.jsonl",
    "trusted/grants.jsonl",
    "private/citations.jsonl",
}


def group_checks(groups, subjects, requests):
    require(len(groups) == 2293, "Pair count")
    anchors = set()
    coverage = Counter()
    subjects_seen = set()
    for pair, members in groups.items():
        require(len(members) == 2, f"Incomplete contrast: {pair}")
        anchor_ids = {m["anchor_request_id"] for m in members}
        profiles = {m["profile"] for m in members}
        require(len(anchor_ids) == 1 and not anchors.intersection(anchor_ids), pair)
        require(len(profiles) == 2 and "valid" in profiles, pair)
        contrast = next(iter(profiles - {"valid"}))
        require(
            all(m["change_dimension"] == DIMENSIONS[contrast] for m in members), pair
        )
        row = requests[members[0]["request_id"]]
        subjects_seen.add(row["subject_id"])
        require(members[0]["evidence"] == members[1]["evidence"], pair)
        anchors.update(anchor_ids)
        coverage[contrast] += 1
    require(
        set(coverage) == NEGATIVES
        and max(coverage.values()) - min(coverage.values()) <= 1,
        "Contrast coverage",
    )
    held_out = {
        uid
        for uid, s in subjects.items()
        if s["split"] == "test" and s["dataset"] == "ltqa"
    }
    require(subjects_seen == held_out and len(held_out) == 14, "Test patient coverage")
    return coverage


def verify(data, raw=None):
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from perm2ask.benchmark import Benchmark
    from perm2ask.notifications import resolve

    verify_release_files(data)
    benchmark = Benchmark(data)
    audit = read_json(data / "audit.json")
    subjects = index(read_rows(data / "benchmark/subjects.jsonl"), "subject_id")
    provenance = index(read_rows(data / "private/provenance.jsonl"), "fact_id")
    facts = check_subjects(subjects, provenance, read_json(data / "splits.json"))
    upstream = raw_profiles(raw)
    policy_data = read_json(data / "policy.json")
    scenarios = index(policy_data["scenarios"], "id")
    policy = {f["id"]: dict(zip(scenarios, f["labels"])) for f in policy_data["fields"]}
    rows = read_rows(data / "benchmark/requests.jsonl")
    check_counts(rows, "test")
    requests = index(rows, "request_id")
    anchors = index(rows[: BASE_ROWS["test"]], "request_id")
    gold = index(read_rows(data / "private/gold.jsonl"), "request_id")
    require(gold.keys() == requests.keys(), "Request/gold coverage")
    additions = index(rows[BASE_ROWS["test"] :], "request_id")
    citations = index(read_rows(data / "private/citations.jsonl"), "request_id")
    require(
        len(additions) == 4586 and citations.keys() == additions.keys(),
        "Test citation coverage",
    )
    channels = {
        "notifications": index(
            read_rows(data / "trusted/notifications.jsonl"), "notification_id"
        ),
        "grants": index(read_rows(data / "trusted/grants.jsonl"), "grant_id"),
    }
    require(
        all(len(rows) == 4730 for rows in channels.values()), "Credential cohort size"
    )
    require(
        all(
            set(r) == {"notification_id", "token"}
            for r in channels["notifications"].values()
        ),
        "Public notification schema",
    )
    require(
        all(set(r) == {"grant_id", "token"} for r in channels["grants"].values()),
        "Public grant schema",
    )
    used_notices, used_grants = set(), set()

    def verify_access(row, trace, fields):
        nid, gid = row["notification_id"], trace["grant_id"]
        require(nid not in used_notices and gid not in used_grants, "Reused credential")
        used_notices.add(nid)
        used_grants.add(gid)
        scenario = scenarios[row["scenario_id"]]
        limits = {f: labels[row["scenario_id"]] for f, labels in policy.items()}
        ceiling = {f for f, v in limits.items() if v}
        actor, active, scope = check_channel_pair(
            row,
            trace,
            channels["notifications"][nid],
            channels["grants"][gid],
            scenario,
            fields,
            ceiling,
        )
        resolved = resolve(row, scenario, limits, channels)
        rid = row.get("request_id") or row["instance_id"]
        require(
            resolved is not None
            and resolved.actor
            == ("fixture-app-" + row["scenario_id"] if actor else None),
            rid,
        )
        require(
            resolved.grant_active == active
            and resolved.fields == scope
            and resolved.focus == row.get("focus"),
            rid,
        )
        subject = subjects[row["subject_id"]]
        allowed_ids = {f["fact_id"] for f in subject["facts"] if f["field_id"] in scope}
        if "confirmed_scope" in subject:
            allowed_ids.intersection_update(
                subject["confirmed_scope"].get(scenario["id"], [])
            )
        allowed_ids.difference_update(
            subject.get("denied_by_scenario", {}).get(scenario["id"], [])
        )
        case = benchmark.case_for(row)
        require(set(case["authorized_ids"]) == allowed_ids, f"Selector scope: {rid}")
        require(
            case["authorization_digest"] == resolved.digest,
            f"Selector cache scope: {rid}",
        )
        return scope

    groups = defaultdict(list)
    anchor_types = Counter()
    for rid, row in additions.items():
        trace = citations[rid]
        anchor_id = trace["anchor_request_id"]
        anchor = anchors[anchor_id]
        require(
            set(row) == set(anchor) | {"notification_id", "pair_id"}
            and rid.startswith("notify-"),
            f"Unexpected public fields: {rid}",
        )
        fields = check_anchor(
            row, gold[rid], anchor, gold[anchor_id], subjects, facts, provenance, policy
        )
        check_citation(trace, gold[rid], provenance, upstream)
        scope = verify_access(row, trace, fields)
        require(
            (fields <= scope) == gold[rid]["allowed"] == (trace["profile"] == "valid"),
            rid,
        )
        require(
            gold[rid]["allowed"] or not fields.intersection(scope),
            f"Mixed permissions: {rid}",
        )
        groups[row["pair_id"]].append(trace)
        if trace["profile"] == "valid":
            anchor_types[
                "v4_abstract"
                if gold[anchor_id].get("derivation") == "intake_abstract"
                else "v3"
            ] += 1
    contrasts = group_checks(groups, subjects, requests)
    require(
        audit["notification_test_profiles"] == {"valid": len(groups), **contrasts},
        "Audit profile counts",
    )
    require(audit["notification_test_pairs"] == len(groups), "Audit pair count")
    require(
        audit["request_count"] == len(requests)
        and audit["verified_facts"] == len(facts),
        "Audit row counts",
    )
    require(
        audit["subject_count"] == len(subjects)
        and audit["test_subjects"]
        == sum(s["split"] == "test" for s in subjects.values()),
        "Audit subject counts",
    )
    audit_counts = index(audit["counts"], "dataset")
    require(audit_counts.keys() == TARGETS["test"].keys(), "Audit datasets")
    observed = Counter(
        (row["dataset"], gold[rid]["allowed"]) for rid, row in requests.items()
    )
    for dataset, totals in audit_counts.items():
        require(
            totals["split"] == "test"
            and totals["allowed"] == observed[dataset, True]
            and totals["denied"] == observed[dataset, False],
            f"Audit decisions: {dataset}",
        )
    rank_summary = index(audit["ranker_splits"], "split")
    require(rank_summary.keys() == set(SPLITS[1:]), "Audit ranker partitions")
    require(
        anchor_types == {"v3": 1210, "v4_abstract": 1083}, "Unexpected anchor selection"
    )
    ids = set(requests)
    pair_ids = set(groups)
    for split in SPLITS[1:]:
        ranking = read_rows(data / f"private/ranker_{split}.jsonl")
        summary = rank_summary[split]
        require(
            summary["instances"] == len(ranking)
            and summary["by_dataset"] == TARGETS[split],
            f"Audit ranking counts: {split}",
        )
        require(
            summary["subjects_with_valid_pairs"]
            == len({r["subject_id"] for r in ranking}),
            f"Audit ranking subjects: {split}",
        )
        rank_ids = index(ranking, "instance_id")
        require(not ids.intersection(rank_ids), "Repeated ID between partitions")
        ids.update(rank_ids)
        cohort = check_ranker(ranking, split, subjects, provenance, facts, policy)
        expected_subjects = {
            uid
            for uid, s in subjects.items()
            if s["dataset"] == "ltqa" and s["split"] == split
        }
        require(cohort == expected_subjects, f"LTQA subject coverage: {split}")
        for row in ranking[BASE_ROWS[split] :]:
            rid, nid, pair = row["instance_id"], row["notification_id"], row["pair_id"]
            require(
                rid.startswith("notify-rank-")
                and "request_id" not in row
                and pair not in pair_ids,
                rid,
            )
            pair_ids.add(pair)
            notice, _ = read_channel(
                channels["notifications"][nid]["token"],
                "fixture-notifier",
                PUBLIC_KEYS["fixture-notifier"],
            )
            trace = {
                "notification_id": nid,
                "pair_id": pair,
                "grant_id": notice["grant_id"],
                "profile": "valid",
            }
            expected_scope = {
                facts[i]["field_id"] for i in row["positive_ids"] + row["negative_ids"]
            }
            scope = verify_access(row, trace, expected_scope)
            require(expected_scope == scope, rid)
    require(
        used_notices == channels["notifications"].keys()
        and used_grants == channels["grants"].keys(),
        "Orphan credentials",
    )
    return {
        "verified": True,
        "v4_files_verified": 8,
        "original_profile_blob_verified": upstream is not None,
        "counts": TARGETS,
        "notification_pairs": len(groups),
        "credentials_per_channel": len(used_notices),
        "test_citations": len(citations),
        "contrasts": dict(sorted(contrasts.items())),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify simulated notification cohorts against the v4 baseline."
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--raw", type=Path, help="Root of the pinned source files")
    args = parser.parse_args()
    print(json.dumps(verify(args.data, args.raw), indent=2))
