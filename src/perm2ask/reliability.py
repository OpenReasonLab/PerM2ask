import re
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from typing import Any

_IDK = re.compile(r"\bi\s+(?:do\s+not|don['’]t)\s+know\b", re.IGNORECASE)


def is_idk_answer(text: str) -> bool:
    normalized = str(text).strip()
    return not normalized or _IDK.search(normalized) is not None


def nli_premise_hypothesis(
    question: str, first_answer: str, second_answer: str
) -> tuple[str, str]:

    premise = f"The answer to the question: {question}\nis {first_answer}."
    hypothesis = f"The answer to the question: {question}\nis {second_answer}."
    return premise, hypothesis


def contradiction_label_index(model_or_config: Any) -> int:
    config = getattr(model_or_config, "config", model_or_config)
    label2id = getattr(config, "label2id", None)
    if label2id is None and isinstance(config, Mapping):
        label2id = config.get("label2id")
    matches: set[int] = set()
    if isinstance(label2id, Mapping):
        for label, index in label2id.items():
            if str(label).casefold() == "contradiction":
                matches.add(int(index))
    id2label = getattr(config, "id2label", None)
    if id2label is None and isinstance(config, Mapping):
        id2label = config.get("id2label")
    if isinstance(id2label, Mapping):
        for index, label in id2label.items():
            if str(label).casefold() == "contradiction":
                matches.add(int(index))
    if len(matches) != 1:
        raise ValueError(
            f"NLI config must identify one contradiction label: {sorted(matches)}"
        )
    return next(iter(matches))


def exact_rank_aware_mis(
    vertices: Sequence[int], edges: Iterable[tuple[int, int]]
) -> list[int]:

    ordered = sorted(set(map(int, vertices)))
    if len(ordered) != len(vertices) or any(vertex < 0 for vertex in ordered):
        raise ValueError("MIS vertices must be unique non-negative ranks")
    allowed = set(ordered)
    adjacency = {vertex: set() for vertex in ordered}
    for raw_left, raw_right in edges:
        left, right = int(raw_left), int(raw_right)
        if left == right or left not in allowed or right not in allowed:
            raise ValueError(f"invalid contradiction edge: {(left, right)}")
        adjacency[left].add(right)
        adjacency[right].add(left)
    for size in range(len(ordered), 0, -1):
        for subset in combinations(ordered, size):
            if all(
                right not in adjacency[left] for left, right in combinations(subset, 2)
            ):
                return list(subset)
    return []


def official_mis_selection(
    candidate_count: int,
    eligible_indices: Sequence[int],
    edges: Iterable[tuple[int, int]],
) -> tuple[list[int], str]:
    if candidate_count < 0:
        raise ValueError("candidate_count must be non-negative")
    eligible = sorted(set(map(int, eligible_indices)))
    if len(eligible) != len(eligible_indices) or any(
        index < 0 or index >= candidate_count for index in eligible
    ):
        raise ValueError("eligible candidate indices are invalid")
    edge_rows = list(edges)
    if not eligible:
        return list(range(candidate_count)), "all_idk_official_fallback"
    selected = exact_rank_aware_mis(eligible, edge_rows)
    if not selected:
        return eligible, "empty_mis_official_fallback"
    return selected, "exact_rank_aware_mis"
