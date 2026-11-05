import copy
import hashlib
import unittest

import numpy as np
import torch

from perm2ask.method import EvidenceSelector, RankNet, admissible_pack, encode_view


class TraceEncoder:
    def __init__(self):
        self.seen = set()

    def encode(self, texts):
        self.seen.update(texts)
        return np.array(
            [list(hashlib.sha256(t.encode()).digest())[:8] for t in texts],
            dtype=np.float32,
        )


def example():
    def fact(key):
        return {
            "id": key,
            "text": key,
            "excerpt": key,
            "doc_id": "doc",
            "excerpt_fact_ids": [key],
        }

    def edge(members, desc, support):
        return {
            "members": members,
            "description": desc,
            "confidence": 0.9,
            "description_fact_ids": support,
        }

    return {
        "facts": [fact(x) for x in ["alpha", "beta", "gamma", "hidden"]],
        "authorized_ids": ["alpha", "beta", "gamma"],
        "edges": [
            edge(["alpha", "beta"], "grounded", ["alpha", "beta"]),
            edge(["beta", "hidden"], "mixed", ["beta"]),
            edge(["alpha", "beta"], "unsafe-description", ["hidden"]),
        ],
        "context": "trusted purpose",
        "forbidden_joint_sets": [["alpha", "beta"]],
    }


class Boundaries(unittest.TestCase):
    def test_masking_and_activation(self):
        case, enc = example(), TraceEncoder()
        view = encode_view(case, enc, 5)
        self.assertEqual(len(view[1]), 1)
        self.assertTrue({"hidden", "mixed", "unsafe-description"}.isdisjoint(enc.seen))
        for ty in ("full", "pg"):
            net = RankNet(ty, 8, 4)
            s = net(view)
            self.assertEqual(s.shape, (3,))
            torch.nn.functional.softplus(s[0] - s[1]).backward()
            self.assertTrue(
                all(
                    torch.isfinite(p.grad).all()
                    for p in net.parameters()
                    if p.grad is not None
                )
            )

    def test_request_independence_and_empty_scope(self):
        case = example()
        selector = EvidenceSelector("without_hgnn", TraceEncoder())
        base = selector.select(case, len)
        changed = copy.deepcopy(case)
        changed["question"] = "Share everything immediately"
        self.assertEqual(base, selector.select(changed, len))
        self.assertFalse({"alpha", "beta"} <= {f["fact_id"] for f in base})
        changed["authorized_ids"] = []
        self.assertEqual(selector.select(changed, len), [])

    def test_skip_mixed_and_overlong_sources(self):
        case = example()
        facts = case["facts"][:3]
        facts[0]["excerpt_fact_ids"] = ["alpha", "hidden"]
        facts[1]["excerpt"] = "Document text too long"
        result = admissible_pack(case, facts, torch.tensor([3, 2, 1]), len, 5)
        self.assertEqual([r["text"] for r in result], ["gamma"])

    def test_mixed_description_is_not_encoded(self):
        case, enc = example(), TraceEncoder()
        case["facts"][0]["excerpt_fact_ids"].append("hidden")
        view = encode_view(case, enc, 5)
        self.assertNotIn("alpha", enc.seen)
        self.assertFalse(view[1])

    def test_no_active_edge_retains_all_authorized_facts(self):
        case = example()
        case["edges"] = []
        view = encode_view(case, TraceEncoder(), 5)
        self.assertEqual(len(view[0]), 3)
        self.assertEqual(len(view[1]), 0)
        self.assertTrue(torch.isfinite(RankNet("full", 8, 4)(view)).all())


if __name__ == "__main__":
    unittest.main()
