from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.text}".strip()


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


@dataclass(frozen=True)
class Bid:
    doc_index: int
    doc_id: str
    agent_id: int
    local_rank: int
    raw_score: float
    calibrated_score: float
    margin_score: float
    standardized_score: float
    lexical_coverage: float
    covered_terms: frozenset[str]
    query_term_count: int


@dataclass
class Dataset:
    documents: list[Document]
    queries: dict[str, Query]
    qrels_train: dict[str, set[str]]
    qrels_test: dict[str, set[str]]


@dataclass
class RetrievalIndex:
    word_vectorizer: TfidfVectorizer
    word_matrix: sparse.csr_matrix
    char_vectorizer: TfidfVectorizer
    char_matrix: sparse.csr_matrix


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} is not valid JSON") from exc


def _read_qrels(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if float(row["score"]) > 0:
                result.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))
    return result


def _load_split_qrels(data_dir: Path, split: str) -> dict[str, set[str]]:
    """Support both the bundled snapshot and the original BEIR qrels layout."""
    candidates = (
        data_dir / f"qrels-{split}.tsv",
        data_dir / "qrels" / f"{split}.tsv",
    )
    for path in candidates:
        if path.exists():
            return _read_qrels(path)
    return {}


def load_dataset(data_dir: Path) -> Dataset:
    documents = [
        Document(str(row["_id"]), row.get("title", ""), row.get("text", ""))
        for row in _read_jsonl(data_dir / "corpus.jsonl")
    ]
    queries = {
        str(row["_id"]): Query(str(row["_id"]), row["text"])
        for row in _read_jsonl(data_dir / "queries.jsonl")
    }
    return Dataset(
        documents=documents,
        queries=queries,
        qrels_train=_load_split_qrels(data_dir, "train"),
        qrels_test=_load_split_qrels(data_dir, "test"),
    )


def build_index(documents: Sequence[Document]) -> RetrievalIndex:
    texts = [doc.full_text for doc in documents]
    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=60_000,
        dtype=np.float32,
    )
    word_matrix = word_vectorizer.fit_transform(texts).tocsr()
    char_vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=50_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char_matrix = char_vectorizer.fit_transform(texts).tocsr()
    return RetrievalIndex(word_vectorizer, word_matrix, char_vectorizer, char_matrix)


def stable_owner(doc_id: str, seed: int, n_agents: int) -> int:
    digest = hashlib.blake2b(f"{seed}:{doc_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % n_agents


def make_shards(
    documents: Sequence[Document],
    n_agents: int,
    seed: int,
    imbalance: float = 0.0,
) -> list[np.ndarray]:
    if imbalance <= 0:
        owners = np.array([stable_owner(doc.doc_id, seed, n_agents) for doc in documents])
    else:
        rng = np.random.default_rng(seed)
        weights = np.arange(1, n_agents + 1, dtype=float) ** (-imbalance)
        weights /= weights.sum()
        owners = rng.choice(n_agents, size=len(documents), p=weights)
    shards = [np.flatnonzero(owners == agent_id) for agent_id in range(n_agents)]
    if any(len(shard) == 0 for shard in shards):
        raise ValueError("At least one shard is empty; adjust n_agents or imbalance")
    return shards


def tokenize(text: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(text.lower()) if len(token) > 2}


def _top_indices(scores: np.ndarray, limit: int) -> np.ndarray:
    limit = min(limit, scores.size)
    if limit == scores.size:
        return np.argsort(-scores, kind="stable")
    chosen = np.argpartition(-scores, limit - 1)[:limit]
    return chosen[np.argsort(-scores[chosen], kind="stable")]


def create_bids(
    query: Query,
    documents: Sequence[Document],
    index: RetrievalIndex,
    shards: Sequence[np.ndarray],
    bid_limit: int,
    heterogeneous: bool,
    corrupt_agent: int | None,
    seed: int,
) -> list[Bid]:
    query_terms = tokenize(query.text)
    word_q = index.word_vectorizer.transform([query.text])
    char_q = index.char_vectorizer.transform([query.text])
    bids: list[Bid] = []
    rng = np.random.default_rng(seed + int(query.query_id) * 1009)

    for agent_id, shard in enumerate(shards):
        use_char = heterogeneous and agent_id % 2 == 1
        matrix = index.char_matrix if use_char else index.word_matrix
        qvec = char_q if use_char else word_q
        scores = (matrix[shard] @ qvec.T).toarray().ravel()
        ranked_local = _top_indices(scores, bid_limit)
        selected_doc_indices = shard[ranked_local]
        selected_scores = scores[ranked_local]

        if corrupt_agent == agent_id:
            # The faulty agent samples low-relevance local items and reports inflated scores.
            low_pool = np.argsort(scores, kind="stable")[: max(bid_limit * 8, bid_limit)]
            ranked_local = rng.choice(low_pool, size=min(bid_limit, len(low_pool)), replace=False)
            selected_doc_indices = shard[ranked_local]
            selected_scores = np.linspace(1.2, 0.9, len(selected_doc_indices))

        score_max = float(np.max(selected_scores)) if len(selected_scores) else 0.0
        score_min = float(np.min(selected_scores)) if len(selected_scores) else 0.0
        scale = max(score_max - score_min, 1e-8)
        score_mean = float(np.mean(selected_scores)) if len(selected_scores) else 0.0
        score_std = max(float(np.std(selected_scores)), 1e-8)

        for local_rank, (doc_index, score) in enumerate(
            zip(selected_doc_indices, selected_scores), start=1
        ):
            doc = documents[int(doc_index)]
            covered = frozenset(query_terms & tokenize(doc.full_text))
            lexical_coverage = len(covered) / max(len(query_terms), 1)
            calibrated = 1.0 - (local_rank - 1) / max(len(selected_doc_indices), 1)
            margin = (float(score) - score_min) / scale
            bids.append(
                Bid(
                    doc_index=int(doc_index),
                    doc_id=doc.doc_id,
                    agent_id=agent_id,
                    local_rank=local_rank,
                    raw_score=float(score),
                    calibrated_score=float(calibrated),
                    margin_score=float(margin),
                    standardized_score=float((float(score) - score_mean) / score_std),
                    lexical_coverage=float(lexical_coverage),
                    covered_terms=covered,
                    query_term_count=max(len(query_terms), 1),
                )
            )
    return bids


def _deduplicate_ordered(bids: Iterable[Bid], budget: int) -> list[Bid]:
    selected: list[Bid] = []
    seen: set[str] = set()
    for bid in bids:
        if bid.doc_id not in seen:
            selected.append(bid)
            seen.add(bid.doc_id)
        if len(selected) == budget:
            break
    return selected


def select_random(bids: Sequence[Bid], budget: int, seed: int) -> list[Bid]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(bids))
    return _deduplicate_ordered((bids[int(i)] for i in order), budget)


def select_round_robin(bids: Sequence[Bid], budget: int) -> list[Bid]:
    return _deduplicate_ordered(
        sorted(bids, key=lambda b: (b.local_rank, b.agent_id, -b.raw_score)), budget
    )


def select_raw_score(bids: Sequence[Bid], budget: int) -> list[Bid]:
    return _deduplicate_ordered(sorted(bids, key=lambda b: -b.raw_score), budget)


def select_minmax_score(bids: Sequence[Bid], budget: int) -> list[Bid]:
    """Merge locally min--max-normalized scores without global collection statistics."""
    return _deduplicate_ordered(
        sorted(
            bids,
            key=lambda b: (-b.margin_score, b.local_rank, b.agent_id, b.doc_id),
        ),
        budget,
    )


def select_zscore_score(bids: Sequence[Bid], budget: int) -> list[Bid]:
    """Merge scores standardized within each emitted local candidate list."""
    return _deduplicate_ordered(
        sorted(
            bids,
            key=lambda b: (-b.standardized_score, b.local_rank, b.agent_id, b.doc_id),
        ),
        budget,
    )


def select_rrf(bids: Sequence[Bid], budget: int, k: int = 60) -> list[Bid]:
    return _deduplicate_ordered(
        sorted(bids, key=lambda b: (-(1.0 / (k + b.local_rank)), -b.raw_score)), budget
    )


def select_scirelay(
    bids: Sequence[Bid],
    budget: int,
    rank_weight: float = 0.35,
    lexical_weight: float = 0.29,
    margin_weight: float = 0.0,
    coverage_weight: float = 0.16,
    diversity_weight: float = 0.0,
) -> list[Bid]:
    remaining = list(bids)
    selected: list[Bid] = []
    seen_docs: set[str] = set()
    covered_terms: set[str] = set()
    agent_counts: dict[int, int] = {}

    while remaining and len(selected) < budget:
        best: Bid | None = None
        best_value = -math.inf
        for bid in remaining:
            if bid.doc_id in seen_docs:
                continue
            novel_terms = len(set(bid.covered_terms) - covered_terms)
            term_gain = novel_terms / bid.query_term_count
            count = agent_counts.get(bid.agent_id, 0)
            diversity_gain = math.sqrt(count + 1.0) - math.sqrt(count)
            value = (
                rank_weight * bid.calibrated_score
                + lexical_weight * bid.lexical_coverage
                + margin_weight * bid.margin_score
                + coverage_weight * term_gain
                + diversity_weight * diversity_gain
            )
            if value > best_value or (
                math.isclose(value, best_value) and best is not None and bid.doc_id < best.doc_id
            ):
                best = bid
                best_value = value
        if best is None:
            break
        selected.append(best)
        seen_docs.add(best.doc_id)
        covered_terms.update(best.covered_terms)
        agent_counts[best.agent_id] = agent_counts.get(best.agent_id, 0) + 1
        remaining.remove(best)
    return selected


def select_centralized(
    query: Query,
    documents: Sequence[Document],
    index: RetrievalIndex,
    budget: int,
) -> list[str]:
    qvec = index.word_vectorizer.transform([query.text])
    scores = (index.word_matrix @ qvec.T).toarray().ravel()
    return [documents[int(i)].doc_id for i in _top_indices(scores, budget)]


def retrieval_metrics(ranked_doc_ids: Sequence[str], relevant: set[str]) -> dict[str, float]:
    hits = np.array([1.0 if doc_id in relevant else 0.0 for doc_id in ranked_doc_ids])
    recall = float(hits.sum() / max(len(relevant), 1))
    precision = float(hits.mean()) if len(hits) else 0.0
    hit_rate = float(hits.max()) if len(hits) else 0.0
    relevant_positions = np.flatnonzero(hits)
    reciprocal_rank = 1.0 / (int(relevant_positions[0]) + 1) if len(relevant_positions) else 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
    dcg = float((hits * discounts).sum())
    ideal_hits = np.ones(min(len(relevant), len(hits)))
    ideal_dcg = float((ideal_hits * discounts[: len(ideal_hits)]).sum())
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "hit_rate": hit_rate,
        "mrr": reciprocal_rank,
        "ndcg": ndcg,
    }


def run_benchmark(
    dataset: Dataset,
    index: RetrievalIndex,
    split: str,
    n_agents: int,
    budgets: Sequence[int],
    seeds: Sequence[int],
    bid_limit: int,
    scenario: str,
    method_weights: Mapping[str, float] | None = None,
    query_limit: int | None = None,
    include_ablations: bool = False,
    policies: Sequence[str] | None = None,
) -> pd.DataFrame:
    qrels = dataset.qrels_test if split == "test" else dataset.qrels_train
    query_ids = sorted(qrels, key=int)
    if query_limit is not None:
        query_ids = query_ids[:query_limit]
    records: list[dict] = []
    doc_index_by_id = {doc.doc_id: i for i, doc in enumerate(dataset.documents)}

    heterogeneous = scenario in {"heterogeneous", "faulty", "imbalanced"}
    imbalance = 1.2 if scenario == "imbalanced" else 0.0
    base_weights = {
        "rank_weight": 0.35,
        "lexical_weight": 0.29,
        "coverage_weight": 0.16,
        "margin_weight": 0.0,
        "diversity_weight": 0.0,
    }
    if method_weights is not None:
        aliases = {
            "rank": "rank_weight",
            "lexical": "lexical_weight",
            "coverage": "coverage_weight",
            "margin": "margin_weight",
            "diversity": "diversity_weight",
        }
        for key, value in method_weights.items():
            target = aliases.get(key, key)
            if target not in base_weights:
                raise ValueError(f"Unknown SciRelay weight: {key}")
            base_weights[target] = float(value)
    if any(value < 0 for value in base_weights.values()):
        raise ValueError("SciRelay weights must be non-negative")

    for seed in seeds:
        shards = make_shards(dataset.documents, n_agents, seed, imbalance=imbalance)
        for query_id in query_ids:
            query = dataset.queries[query_id]
            corrupt_agent = 0 if scenario == "faulty" else None
            bids = create_bids(
                query,
                dataset.documents,
                index,
                shards,
                bid_limit,
                heterogeneous,
                corrupt_agent,
                seed,
            )
            for budget in budgets:
                requested_policies = set(policies) if policies is not None else None
                wants = lambda name: requested_policies is None or name in requested_policies
                selected_by_policy: dict[str, list[Bid]] = {}
                if wants("random"):
                    selected_by_policy["random"] = select_random(
                        bids, budget, seed + int(query_id)
                    )
                if wants("round_robin"):
                    selected_by_policy["round_robin"] = select_round_robin(bids, budget)
                if wants("raw_score"):
                    selected_by_policy["raw_score"] = select_raw_score(bids, budget)
                if wants("minmax_score"):
                    selected_by_policy["minmax_score"] = select_minmax_score(bids, budget)
                if wants("zscore_score"):
                    selected_by_policy["zscore_score"] = select_zscore_score(bids, budget)
                if wants("rrf"):
                    selected_by_policy["rrf"] = select_rrf(bids, budget)
                if wants("scirelay"):
                    selected_by_policy["scirelay"] = select_scirelay(
                        bids, budget, **base_weights
                    )
                if include_ablations:
                    selected_by_policy.update(
                        {
                            "scirelay_with_raw_margin": select_scirelay(
                                bids, budget, **{**base_weights, "margin_weight": 0.20}
                            ),
                            "scirelay_no_coverage": select_scirelay(
                                bids,
                                budget,
                                **{
                                    **base_weights,
                                    "lexical_weight": 0.0,
                                    "coverage_weight": 0.0,
                                },
                            ),
                            "scirelay_with_diversity": select_scirelay(
                                bids, budget, **{**base_weights, "diversity_weight": 0.08}
                            ),
                            "scirelay_no_calibration": select_scirelay(
                                bids, budget, **{**base_weights, "rank_weight": 0.0}
                            ),
                        }
                    )

                for policy, selected in selected_by_policy.items():
                    ranked_doc_ids = [bid.doc_id for bid in selected]
                    metrics = retrieval_metrics(ranked_doc_ids, qrels[query_id])
                    payload_bytes = sum(
                        len(dataset.documents[bid.doc_index].full_text.encode("utf-8"))
                        for bid in selected
                    )
                    control_width = {
                        "raw_score": 12,
                        "minmax_score": 10,
                        "zscore_score": 10,
                        "rrf": 9,
                    }.get(policy, 20 if policy.startswith("scirelay") else 0)
                    records.append(
                        {
                            "split": split,
                            "scenario": scenario,
                            "seed": seed,
                            "query_id": query_id,
                            "n_agents": n_agents,
                            "bid_limit": bid_limit,
                            "budget": budget,
                            "policy": policy,
                            "payload_bytes": payload_bytes,
                            "control_bytes": len(bids) * control_width,
                            "total_bytes": payload_bytes + len(bids) * control_width,
                            **metrics,
                        }
                    )
                if requested_policies is None or "centralized_word" in requested_policies:
                    centralized_ids = select_centralized(query, dataset.documents, index, budget)
                    centralized_payload = sum(
                        len(dataset.documents[doc_index_by_id[doc_id]].full_text.encode("utf-8"))
                        for doc_id in centralized_ids
                    )
                    records.append(
                        {
                            "split": split,
                            "scenario": scenario,
                            "seed": seed,
                            "query_id": query_id,
                            "n_agents": n_agents,
                            "bid_limit": bid_limit,
                            "budget": budget,
                            "policy": "centralized_word",
                            "payload_bytes": centralized_payload,
                            "control_bytes": 0,
                            "total_bytes": centralized_payload,
                            **retrieval_metrics(centralized_ids, qrels[query_id]),
                        }
                    )
    return pd.DataFrame.from_records(records)
