import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

INSTANT = 1793836800
AUDIENCE = "perm2ask-simulated-runtime"
NOTIFIER = "fixture-notifier"
GRANTS = "fixture-grants"
PUBLIC_KEYS = {
    NOTIFIER: "df7f7a3565edf7ee1f95ec901643f36572d52fb1b7777987d9ab9f2c228cc7eb",
    GRANTS: "40bc77b80c88ae6852ae83ca2e28cc5ccf117d9a3a66aa0af1e4ba8850f161cb",
}
FILES = {
    "notifications": "trusted/notifications.jsonl",
    "grants": "trusted/grants.jsonl",
}


def encoded(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def decoded(segment):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", segment, re.ASCII):
        raise ValueError("Invalid compact JWS encoding")
    raw = base64.b64decode(
        segment + "=" * (-len(segment) % 4), altchars=b"-_", validate=True
    )
    if encoded(raw) != segment:
        raise ValueError("Noncanonical compact JWS encoding")
    return raw


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def strict_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("Duplicate signed claim")
        obj[key] = value
    return obj


def fixture_token(issuer, claims):
    header = {"alg": "EdDSA", "kid": issuer, "typ": "JWT"}
    message = encoded(canonical(header)) + "." + encoded(canonical(claims))
    seed = hashlib.sha256(
        ("PerM2ask synthetic fixture key: " + issuer).encode()
    ).digest()
    signature = Ed25519PrivateKey.from_private_bytes(seed).sign(message.encode("ascii"))
    return message + "." + encoded(signature)


def verified_token(token, issuer):
    if not isinstance(token, str) or len(token) > 16384:
        raise ValueError("Invalid compact JWS")
    header, payload, signature = token.split(".")
    if json.loads(decoded(header).decode("utf8"), object_pairs_hook=strict_object) != {
        "alg": "EdDSA",
        "kid": issuer,
        "typ": "JWT",
    }:
        raise ValueError("Unexpected JWS header or issuer")
    Ed25519PublicKey.from_public_bytes(bytes.fromhex(PUBLIC_KEYS[issuer])).verify(
        decoded(signature), (header + "." + payload).encode("ascii")
    )
    obj = json.loads(decoded(payload).decode("utf8"), object_pairs_hook=strict_object)
    if not isinstance(obj, dict):
        raise ValueError("Expected signed object claims")
    return obj


def fixture_channels(request, scenario, field_ids, profile="valid"):
    profiles = {
        "valid",
        "wrong_actor",
        "bad_signature",
        "expired_notification",
        "missing_grant",
        "revoked_grant",
        "narrow_grant",
        "wrong_grant_subject",
    }
    if profile not in profiles:
        raise ValueError("Unknown synthetic fixture profile")
    actor = "fixture-app-" + scenario["id"]
    common = {
        "sub": request["subject_id"],
        "scenario": scenario["id"],
        "actor": actor,
        "role": scenario["role"],
        "purpose": scenario["purpose"],
        "rid": request.get("request_id") or request["instance_id"],
        "aud": AUDIENCE,
        "nbf": INSTANT - 3600,
        "exp": INSTANT + 3600,
    }
    fields = sorted(set(field_ids))
    grant = dict(
        common,
        iss=GRANTS,
        allowed_fields=fields,
        status="revoked" if profile == "revoked_grant" else "active",
    )
    if profile == "wrong_grant_subject":
        grant["sub"] = "other-simulated-subject"
    gid = "grant-" + hashlib.sha256(canonical(grant)).hexdigest()[:24]
    grant["grant_id"] = gid
    nid = request["notification_id"]
    notice = dict(
        common,
        iss=NOTIFIER,
        notification_id=nid,
        pair_id=request["pair_id"],
        grant_id=gid,
    )
    if "instance_id" in request:
        notice["focus"] = request["focus"]
    if profile == "wrong_actor":
        notice["actor"] = "fixture-unregistered-app"
    if profile == "expired_notification":
        notice["exp"] = INSTANT
    token = fixture_token(NOTIFIER, notice)
    if profile == "bad_signature":
        prefix, signature = token.rsplit(".", 1)
        raw = decoded(signature)
        token = prefix + "." + encoded(bytes([raw[0] ^ 1]) + raw[1:])
    return (
        {"notification_id": nid, "token": token},
        None
        if profile == "missing_grant"
        else {"grant_id": gid, "token": fixture_token(GRANTS, grant)},
    )


def notification_case(request):
    return (
        str(request.get("request_id", "")).startswith("notify-")
        or str(request.get("instance_id", "")).startswith("notify-")
        or "notification_id" in request
        or "pair_id" in request
    )


def read_channels(root):
    channels = {}
    for name, rel in FILES.items():
        index = {}
        path = Path(root) / rel
        if path.is_file():
            with path.open(encoding="utf-8-sig") as stream:
                for line in stream:
                    row = json.loads(line)
                    key = row[
                        "notification_id" if name == "notifications" else "grant_id"
                    ]
                    if not isinstance(key, str) or not key or key in index:
                        raise ValueError(f"Invalid or duplicate identifier in {name}")
                    index[key] = row
        channels[name] = index
    return channels


@dataclass(frozen=True)
class Access:
    fields: frozenset
    actor: str | None
    grant_active: bool
    digest: str
    focus: str | None = None

    def descriptor(self):
        return {
            "verified_actor": self.actor,
            "active_patient_grant": self.grant_active,
            "effective_fields": sorted(self.fields),
        }


def bound_claims(token, issuer, expected, extra):
    claims = verified_token(token, issuer)
    if set(claims) != set(expected) | {"iss", "aud", "nbf", "exp"} | set(extra):
        raise ValueError("Unexpected signed claim set")
    if (
        claims["iss"] != issuer
        or claims["aud"] != AUDIENCE
        or any(claims[k] != v for k, v in expected.items())
    ):
        raise ValueError("Signed identity binding mismatch")
    if (
        type(claims["nbf"]) is not int
        or type(claims["exp"]) is not int
        or not claims["nbf"] <= INSTANT < claims["exp"]
    ):
        raise ValueError("Credential is outside its validity interval")
    return claims


def resolve(request, scenario, permissions, channels=None):
    if not notification_case(request):
        return None
    channels = channels or {}
    fields = frozenset()
    actor = None
    active = False
    token = grant_token = focus = None
    try:
        rid = request.get("request_id") or request.get("instance_id")
        nid = request["notification_id"]
        pair_id = request["pair_id"]
        if (
            not all(
                isinstance(v, str) and v
                for v in (nid, pair_id, rid, request["subject_id"])
            )
            or not rid.startswith("notify-")
            or ("instance_id" in request and rid != request["instance_id"])
            or request["scenario_id"] != scenario["id"]
        ):
            raise ValueError("Invalid notification binding")
        expected = {
            "sub": request["subject_id"],
            "scenario": scenario["id"],
            "role": scenario["role"],
            "purpose": scenario["purpose"],
            "rid": rid,
            "actor": "fixture-app-" + scenario["id"],
        }
        notice_binding = dict(expected, notification_id=nid, pair_id=pair_id)
        if "instance_id" in request:
            if (
                not isinstance(request.get("focus"), str)
                or not request["focus"].strip()
            ):
                raise ValueError("Missing signed supervision focus")
            notice_binding["focus"] = request["focus"]
        token = channels["notifications"][nid]["token"]
        notice = bound_claims(
            token,
            NOTIFIER,
            notice_binding,
            {"grant_id"},
        )
        actor = expected["actor"]
        focus = notice.get("focus")
        gid = notice["grant_id"]
        if not isinstance(gid, str) or not gid:
            raise ValueError("Missing patient grant binding")
        grant_token = channels["grants"][gid]["token"]
        grant = bound_claims(
            grant_token,
            GRANTS,
            dict(expected, grant_id=gid),
            {"allowed_fields", "status"},
        )
        scope = grant["allowed_fields"]
        if (
            grant["status"] != "active"
            or not isinstance(scope, list)
            or not all(isinstance(f, str) and f in permissions for f in scope)
            or len(set(scope)) != len(scope)
        ):
            raise ValueError("Invalid or inactive patient scope")
        fields = frozenset(f for f in scope if permissions[f])
        active = True
    except (
        KeyError,
        ValueError,
        TypeError,
        InvalidSignature,
        binascii.Error,
        RecursionError,
    ):
        pass
    fingerprint = hashlib.sha256(
        canonical(
            [
                request.get(k)
                for k in (
                    "request_id",
                    "notification_id",
                    "pair_id",
                    "subject_id",
                    "scenario_id",
                )
            ]
            + [token, grant_token, actor, active, sorted(fields)]
        )
    ).hexdigest()
    return Access(fields, actor, active, fingerprint, focus)


def context_text(scenario, access):
    if access.actor is None:
        return (
            "Application identity could not be verified. No patient scope is available."
        )
    return json.dumps(
        {
            "scenario": scenario["name"],
            "role": scenario["role"],
            "purpose": scenario["purpose"],
            **access.descriptor(),
            **({"focus": access.focus} if access.focus else {}),
        },
        sort_keys=True,
    )
