from scirelay.benchmark import (
    Bid,
    Document,
    Query,
    retrieval_metrics,
    select_minmax_score,
    select_scirelay,
    select_zscore_score,
)
from scirelay.dense import create_mixed_bids

import numpy as np


def make_bid(
    doc_id: str,
    agent: int,
    rank: int,
    coverage: float,
    *,
    raw_score: float | None = None,
    minmax_score: float | None = None,
    zscore: float | None = None,
) -> Bid:
    return Bid(
        doc_index=int(doc_id),
        doc_id=doc_id,
        agent_id=agent,
        local_rank=rank,
        raw_score=coverage if raw_score is None else raw_score,
        calibrated_score=1.0 / rank,
        margin_score=coverage if minmax_score is None else minmax_score,
        standardized_score=coverage if zscore is None else zscore,
        lexical_coverage=coverage,
        covered_terms=frozenset({doc_id}),
        query_term_count=3,
    )


def test_metrics_are_bounded() -> None:
    metrics = retrieval_metrics(["1", "2", "3"], {"2", "4"})
    assert metrics["recall"] == 0.5
    assert metrics["mrr"] == 0.5
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_scirelay_respects_budget_and_deduplicates() -> None:
    bids = [
        make_bid("1", 0, 1, 0.9),
        make_bid("1", 1, 1, 0.8),
        make_bid("2", 1, 2, 0.7),
        make_bid("3", 2, 1, 0.6),
    ]
    selected = select_scirelay(bids, budget=2)
    assert len(selected) == 2
    assert len({bid.doc_id for bid in selected}) == 2


def test_minmax_score_uses_local_normalization() -> None:
    bids = [
        make_bid("1", 0, 1, 0.2, raw_score=100.0, minmax_score=1.0),
        make_bid("2", 0, 2, 0.0, raw_score=90.0, minmax_score=0.0),
        make_bid("3", 1, 1, 1.0, raw_score=1.0, minmax_score=1.0),
    ]
    selected = select_minmax_score(bids, budget=2)
    assert [bid.doc_id for bid in selected] == ["1", "3"]


def test_zscore_score_uses_standardized_value() -> None:
    bids = [
        make_bid("1", 0, 1, 0.2, raw_score=100.0, zscore=-1.0),
        make_bid("2", 1, 1, 0.8, raw_score=1.0, zscore=2.0),
    ]
    selected = select_zscore_score(bids, budget=1)
    assert selected[0].doc_id == "2"


def test_lexical_dense_bids_use_the_declared_agent_retriever() -> None:
    documents = [
        Document(str(index), f"Title {index}", f"Text {index}") for index in range(4)
    ]
    bids = create_mixed_bids(
        query=Query("1", "scientific claim"),
        documents=documents,
        shards=[np.asarray([0, 1]), np.asarray([2, 3])],
        word_scores=np.asarray([0.9, 0.1, 0.8, 0.2]),
        dense_scores=np.asarray([0.1, 0.9, 0.2, 0.8]),
        document_terms=[frozenset() for _ in documents],
        query_terms=frozenset({"scientific", "claim"}),
        bid_limit=1,
        scenario="lexical_dense_heterogeneous",
        seed=11,
    )
    assert [(bid.agent_id, bid.doc_id) for bid in bids] == [(0, "0"), (1, "3")]
