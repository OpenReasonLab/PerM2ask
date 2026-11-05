from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .fetch_bge import BGE_FILES, BGE_REPO, BGE_REVISION


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FrozenBGE:
    def __init__(self, folder):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        folder = Path(folder)
        manifest = json.loads((folder / "source.json").read_text(encoding="utf-8"))
        if manifest.get("repo") != BGE_REPO or manifest.get("revision") != BGE_REVISION:
            raise ValueError("Unexpected BGE encoder revision")
        for name in ("onnx/model.onnx", "tokenizer.json"):
            if (
                manifest["sha256"][name] != BGE_FILES[name]
                or digest(folder / name) != BGE_FILES[name]
            ):
                raise ValueError(f"Encoder hash mismatch: {name}")
        self.signature = (
            manifest["sha256"]["onnx/model.onnx"]
            + ":"
            + manifest["sha256"]["tokenizer.json"]
        )
        self.tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tok.enable_truncation(max_length=512)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(folder / "onnx/model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.dimension = 384
        self._cached_vector = lru_cache(maxsize=16384)(self._one)

    def _one(self, text):
        encoded = self.tok.encode(text)
        inputs = {
            "input_ids": encoded.ids,
            "attention_mask": encoded.attention_mask,
            "token_type_ids": encoded.type_ids,
        }
        feed = {
            item.name: np.asarray([inputs[item.name]], dtype=np.int64)
            for item in self.session.get_inputs()
        }
        result = self.session.run(None, feed)[0]
        vec = result[0, 0] if result.ndim == 3 else result[0]
        return vec / max(np.linalg.norm(vec), 1e-12)

    def encode(self, texts):
        return np.stack([self._cached_vector(text) for text in texts])


def check_ids(values, universe, name):
    if (
        not isinstance(values, list)
        or len(set(values)) != len(values)
        or not set(values) <= universe
    ):
        raise ValueError(f"Unknown or duplicate fact IDs in {name}")
    return set(values)


def encode_view(case, encoder, seed_count):
    facts = case["facts"]
    keyed = {f["id"]: f for f in facts}
    if len(keyed) != len(facts):
        raise ValueError("Duplicate fact ID")
    permitted = check_ids(case["authorized_ids"], keyed.keys(), "authorized_ids")
    visible = []
    for fact in facts:
        if fact["id"] not in permitted:
            continue

        default_scope = (
            fact.get("excerpt_fact_ids", [])
            if fact.get("excerpt") == fact["text"]
            else [fact["id"]]
        )
        scope = check_ids(
            fact.get("text_fact_ids", default_scope), keyed.keys(), "text_fact_ids"
        )
        if fact["id"] in scope and scope <= permitted:
            visible.append(fact)
    if not visible:
        return visible, [], None
    if not isinstance(case["context"], str) or not case["context"].strip():
        raise ValueError("Expected a trusted context string")
    x = F.normalize(
        torch.as_tensor(
            encoder.encode([case["context"]] + [f["text"] for f in visible]),
            dtype=torch.float32,
        ),
        dim=1,
    )
    xc, xv = x[0], x[1:]
    rho = xv @ xc
    rows = {f["id"]: i for i, f in enumerate(visible)}
    seeds = {
        visible[i]["id"]
        for i in torch.argsort(rho, descending=True, stable=True)[:seed_count].tolist()
    }
    active = []
    for edge in case["edges"]:
        members = check_ids(edge["members"], keyed.keys(), "members")
        support = check_ids(
            edge["description_fact_ids"], keyed.keys(), "description_fact_ids"
        )
        if not (
            members & seeds
            and members <= rows.keys()
            and support
            and (members | support) <= permitted
        ):
            continue
        if not isinstance(edge["description"], str) or not edge["description"].strip():
            raise ValueError("Missing association description")
        conf = float(edge["confidence"])
        if not 0 <= conf <= 1:
            raise ValueError("Invalid fixed association confidence")
        active.append(
            ([rows[fid] for fid in edge["members"]], conf, edge["description"])
        )
    xe = None
    if active:
        xe = F.normalize(
            torch.as_tensor(
                encoder.encode([edge[2] for edge in active]), dtype=torch.float32
            ),
            dim=1,
        )
    return visible, active, (xc, xv, xe, rho)


def mlp(input_dim, hidden):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))


class RankNet(nn.Module):
    def __init__(self, method, dimension, hidden):
        super().__init__()
        self.method = method
        self.context = nn.Linear(dimension, hidden, bias=False)
        self.fact = nn.Linear(dimension, hidden, bias=False)
        self.self_loop = nn.Linear(hidden, hidden, bias=False)
        self.out_map = nn.Linear(hidden, hidden, bias=False)
        self.scorer = mlp(2 * hidden + 1, hidden)
        if method == "full":
            self.edge = nn.Linear(dimension, hidden, bias=False)
            self.in_map = nn.Linear(hidden, hidden, bias=False)
            self.g_in = mlp(3 * hidden + 1, hidden)
            self.g_out = mlp(3 * hidden + 1, hidden)
        elif method == "pg":
            self.neighbor_score = mlp(3 * hidden, hidden)
        else:
            raise ValueError(method)

    def forward(self, view):
        facts, edges, (xc, xv, xe, rho) = view
        c, z = self.context(xc), self.fact(xv)
        messages = [[] for _ in facts]
        weights = [[] for _ in facts]
        if self.method == "full":
            for (members, confidence, _), edge in zip(
                edges, self.edge(xe) if edges else []
            ):
                n = len(members)
                local = z[members]
                conf = local.new_full((n, 1), confidence)
                g = self.g_in(
                    torch.cat([c.expand(n, -1), local, edge.expand(n, -1), conf], dim=1)
                ).flatten()
                agg = torch.tanh(self.in_map((g.softmax(0)[:, None] * local).sum(0)))
                incident = self.g_out(
                    torch.cat([c.expand(n, -1), local, agg.expand(n, -1), conf], dim=1)
                ).flatten()
                mapped = self.out_map(agg)
                for node, w in zip(members, incident):
                    weights[node].append(w)
                    messages[node].append(mapped)
        else:
            neighbors = [set() for _ in facts]
            for members, _, _ in edges:
                for node in members:
                    neighbors[node].update(other for other in members if other != node)
            for node, adjacent in enumerate(neighbors):
                for neighbor in sorted(adjacent):
                    weights[node].append(
                        self.neighbor_score(
                            torch.cat((c, z[node], z[neighbor]))
                        ).squeeze(-1)
                    )
                    messages[node].append(self.out_map(z[neighbor]))
        updated = []
        for node, own in enumerate(self.self_loop(z)):
            if messages[node]:
                att = torch.stack(weights[node]).softmax(0)
                own = own + (att[:, None] * torch.stack(messages[node])).sum(0)
            updated.append(torch.tanh(own))
        return self.scorer(
            torch.cat(
                (c.expand(len(facts), -1), torch.stack(updated), rho[:, None]), dim=1
            )
        ).flatten()


def raw_similarity_scores(view):
    _, edges, (xc, _, xe, rho) = view
    edge_scores = defaultdict(list)
    for (members, _, _), cosine in zip(edges, xe @ xc if edges else []):
        for node in members:
            edge_scores[node].append(float(cosine))
    return torch.tensor(
        [float(sim) + max(edge_scores[i], default=0.0) for i, sim in enumerate(rho)]
    )


def admissible_pack(case, facts, scores, count_tokens, budget):
    universe = {fact["id"] for fact in case["facts"]}
    authorized = set(case["authorized_ids"])
    joint = [
        check_ids(values, universe, "forbidden_joint_sets")
        for values in case.get("forbidden_joint_sets", [])
    ]
    chosen, selected_scope, seen = [], set(), set()
    for i in torch.argsort(scores, descending=True, stable=True).tolist():
        f = facts[i]
        excerpt = f.get("excerpt")
        scope = check_ids(f.get("excerpt_fact_ids", []), universe, "excerpt_fact_ids")
        if (
            not excerpt
            or not excerpt.strip()
            or not f.get("doc_id")
            or f["id"] not in scope
            or not scope <= authorized
        ):
            continue
        if excerpt in seen or any(combo <= (scope | selected_scope) for combo in joint):
            continue
        text = "\n".join([item["text"] for item in chosen] + [excerpt])
        if count_tokens(text) > budget:
            continue
        chosen.append({"fact_id": f["id"], "doc_id": f["doc_id"], "text": excerpt})
        selected_scope.update(scope)
        seen.add(excerpt)
    return chosen


class EvidenceSelector:
    def __init__(self, method, encoder, checkpoint=None, seed_count=5, budget=2048):
        if budget < 1 or seed_count < 1 or method not in {"full", "pg", "without_hgnn"}:
            raise ValueError("Invalid ranker parameters")
        self.method = method
        self.encoder = (
            FrozenBGE(encoder) if isinstance(encoder, (str, Path)) else encoder
        )
        self.k, self.budget = seed_count, budget
        self.training_subjects = set()
        if method == "without_hgnn":
            if checkpoint:
                raise ValueError("without_hgnn takes no checkpoint")
            self.net = None
        else:
            if not checkpoint:
                raise ValueError(f"{method} requires its own trained checkpoint")
            ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if (ck["method"], ck["encoder_signature"], ck["k"]) != (
                method,
                self.encoder.signature,
                seed_count,
            ):
                raise ValueError("Checkpoint method, encoder or seed count mismatch")
            self.training_subjects = set(ck["training_subjects"]) | set(
                ck["validation_subjects"]
            )
            self.net = RankNet(method, self.encoder.dimension, ck["hidden"])
            self.net.load_state_dict(ck["state"])
            self.net.eval()

    @torch.no_grad()
    def select(self, instance, count_tokens):
        if (
            instance.get("split") == "test"
            and instance["subject_id"] in self.training_subjects
        ):
            raise ValueError("Test subject participated in training/validation")
        view = encode_view(instance, self.encoder, self.k)
        if not view[0]:
            return []
        scores = self.net(view) if self.net is not None else raw_similarity_scores(view)
        return admissible_pack(instance, view[0], scores, count_tokens, self.budget)
