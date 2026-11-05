import json
from pathlib import Path

from . import notifications


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


class Benchmark:
    def __init__(self, root):
        root = Path(root)
        policy = json.loads((root / "policy.json").read_text(encoding="utf-8-sig"))
        self.fields = {item["id"]: item["labels"] for item in policy["fields"]}
        if any(
            len(labels) != 9 or any(type(v) is not bool for v in labels)
            for labels in self.fields.values()
        ):
            raise ValueError("Expected nine Boolean decisions for each field")
        self.scenarios = {item["id"]: item for item in policy["scenarios"]}
        self.subjects = {}
        for row in read_rows(root / "benchmark/subjects.jsonl"):
            if row["subject_id"] in self.subjects:
                raise ValueError(
                    "Duplicate subject ID or subject reused across partitions"
                )
            self.subjects[row["subject_id"]] = row
        self.requests = root / "benchmark/requests.jsonl"
        self.channels = notifications.read_channels(root)

    def access_for(self, request):
        sid = request["scenario_id"]
        permissions = {
            field: flags[int(sid[1:]) - 1] for field, flags in self.fields.items()
        }
        return notifications.resolve(
            request, self.scenarios[sid], permissions, self.channels
        )

    def case_for(self, request):
        subject = self.subjects[request["subject_id"]]
        if (
            request["dataset"] != subject["dataset"]
            or request["split"] != subject["split"]
        ):
            raise ValueError("Request and subject partitions differ")
        case = self.case(request["subject_id"], request["scenario_id"])
        access = self.access_for(request)
        if access is not None:
            case["authorization_digest"] = access.digest
            permitted = {
                f["id"] for f in case["facts"] if f["field_id"] in access.fields
            }
            case["authorized_ids"] = [
                fid for fid in case["authorized_ids"] if fid in permitted
            ]
            case["context"] = notifications.context_text(
                self.scenarios[request["scenario_id"]], access
            )
        return case

    def case(self, subject_id, scenario_id):
        row = self.subjects[subject_id]
        scenario = self.scenarios[scenario_id]
        index = int(scenario_id.removeprefix("S")) - 1
        if not 0 <= index < 9:
            raise ValueError("Invalid scenario")
        docs = {d["document_id"]: d["text"] for d in row["documents"]}
        authorized, facts = [], []
        for f in row["facts"]:
            text, start, end = f["text"], f["start"], f["end"]
            if (
                not (0 <= start < end <= len(docs[f["document_id"]]))
                or docs[f["document_id"]][start:end] != text
            ):
                raise ValueError(f"Invalid source span {f['fact_id']}")
            if self.fields[f["field_id"]][index]:
                authorized.append(f["fact_id"])
            facts.append(
                {
                    "id": f["fact_id"],
                    "field_id": f["field_id"],
                    "text": text,
                    "doc_id": f["document_id"],
                    "excerpt": text,
                    "excerpt_fact_ids": f.get("excerpt_fact_ids", [f["fact_id"]]),
                }
            )

        if "confirmed_scope" in row:
            scopes = row["confirmed_scope"]
            scope = scopes.get(scenario_id, [])
            if not isinstance(scope, list) or not set(scope) <= {
                f["id"] for f in facts
            }:
                raise ValueError("Unknown fact in confirmed personal scope")
            authorized = [fid for fid in authorized if fid in set(scope)]
        if "denied_by_scenario" in row:
            denied = row["denied_by_scenario"].get(scenario_id, [])
            if not isinstance(denied, list) or not set(denied) <= {
                f["id"] for f in facts
            }:
                raise ValueError("Unknown fact in confirmed personal prohibitions")
            authorized = [fid for fid in authorized if fid not in set(denied)]
        return {
            "subject_id": subject_id,
            "scenario_id": scenario_id,
            "split": row["split"],
            "context": "\n".join(
                f"{k}: {scenario[k]}" for k in ("name", "role", "purpose")
            ),
            "authorized_ids": authorized,
            "facts": facts,
            "edges": row.get("source_associations", []),
            "forbidden_joint_sets": row.get("forbidden_joint_sets", []),
        }
