from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from scirelay.benchmark import Bid, Document, Query


@dataclass(frozen=True)
class DenseIndex:
    """Frozen, L2-normalized SPECTER representations and provenance."""

    document_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    document_matrix: np.ndarray
    query_matrix: np.ndarray
    model_id: str
    revision: str
    pooling: str
    document_format: str

    @property
    def document_position(self) -> dict[str, int]:
        return {doc_id: index for index, doc_id in enumerate(self.document_ids)}

    @property
    def query_position(self) -> dict[str, int]:
        return {query_id: index for index, query_id in enumerate(self.query_ids)}


def load_dense_index(cache_dir: Path, documents: Sequence[Document]) -> DenseIndex:
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    document_ids = tuple(
        str(value) for value in np.load(cache_dir / "document_ids.npy", allow_pickle=False)
    )
    query_ids = tuple(
        str(value) for value in np.load(cache_dir / "query_ids.npy", allow_pickle=False)
    )
    document_matrix = np.load(
        cache_dir / "document_embeddings.npy", mmap_mode="r", allow_pickle=False
    )
    query_matrix = np.load(
        cache_dir / "query_embeddings.npy", mmap_mode="r", allow_pickle=False
    )
    expected_document_ids = tuple(document.doc_id for document in documents)
    if document_ids != expected_document_ids:
        raise ValueError("Dense document order does not match the loaded corpus")
    if document_matrix.shape != (len(document_ids), int(metadata["hidden_size"])):
        raise ValueError("Dense document matrix shape is inconsistent with metadata")
    if query_matrix.shape != (len(query_ids), int(metadata["hidden_size"])):
        raise ValueError("Dense query matrix shape is inconsistent with metadata")
    return DenseIndex(
        document_ids=document_ids,
        query_ids=query_ids,
        document_matrix=document_matrix,
        query_matrix=query_matrix,
        model_id=str(metadata["model_id"]),
        revision=str(metadata["revision"]),
        pooling=str(metadata["pooling"]),
        document_format=str(metadata["document_format"]),
    )


def _top_indices(scores: np.ndarray, limit: int) -> np.ndarray:
    limit = min(limit, scores.size)
    if limit <= 0:
        return np.empty(0, dtype=int)
    if limit == scores.size:
        return np.argsort(-scores, kind="stable")
    chosen = np.argpartition(-scores, limit - 1)[:limit]
    return chosen[np.argsort(-scores[chosen], kind="stable")]


def create_mixed_bids(
    query: Query,
    documents: Sequence[Document],
    shards: Sequence[np.ndarray],
    word_scores: np.ndarray,
    dense_scores: np.ndarray,
    document_terms: Sequence[frozenset[str]],
    query_terms: frozenset[str],
    bid_limit: int,
    scenario: str,
    seed: int,
) -> list[Bid]:
    """Create a shared candidate pool for dense and lexical--dense scenarios."""

    if scenario not in {
        "dense_homogeneous",
        "lexical_dense_heterogeneous",
        "dense_faulty",
    }:
        raise ValueError(f"Unknown dense scenario: {scenario}")
    rng = np.random.default_rng(seed + int(query.query_id) * 1009)
    bids: list[Bid] = []
    query_term_count = max(len(query_terms), 1)

    for agent_id, shard in enumerate(shards):
        use_dense = scenario != "lexical_dense_heterogeneous" or agent_id % 2 == 1
        source_scores = dense_scores if use_dense else word_scores
        local_scores = np.asarray(source_scores[shard], dtype=np.float64)
        ranked_local = _top_indices(local_scores, bid_limit)
        selected_doc_indices = shard[ranked_local]
        selected_scores = local_scores[ranked_local]

        if scenario == "dense_faulty" and agent_id == 0:
            low_pool = np.argsort(local_scores, kind="stable")[: max(bid_limit * 8, bid_limit)]
            ranked_local = rng.choice(
                low_pool, size=min(bid_limit, len(low_pool)), replace=False
            )
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
            integer_index = int(doc_index)
            covered = frozenset(query_terms & document_terms[integer_index])
            coverage = len(covered) / query_term_count
            calibrated = 1.0 - (local_rank - 1) / max(len(selected_doc_indices), 1)
            bids.append(
                Bid(
                    doc_index=integer_index,
                    doc_id=documents[integer_index].doc_id,
                    agent_id=agent_id,
                    local_rank=local_rank,
                    raw_score=float(score),
                    calibrated_score=float(calibrated),
                    margin_score=float((float(score) - score_min) / scale),
                    standardized_score=float((float(score) - score_mean) / score_std),
                    lexical_coverage=float(coverage),
                    covered_terms=covered,
                    query_term_count=query_term_count,
                )
            )
    return bids
