import argparse
from unittest.mock import MagicMock, patch

import pytest

from src.config import RAGConfig
from src.main import build_query_planner, get_answer, resolve_query_config
from src.planning.cost_based import AdaptiveQueryPlanner, CostBasedQueryPlanner
from src.planning.heuristics import HeuristicQueryPlanner


pytestmark = pytest.mark.unit


def test_heuristic_planner_classifies_queries():
    planner = HeuristicQueryPlanner(RAGConfig())

    assert planner.classify("What is BCNF?") == "definitional"
    assert planner.classify("Why is strict two-phase locking used?") == "explanatory"
    assert planner.classify("Compare OLTP and OLAP workloads.") == "analytical"
    assert planner.classify("How do I build a B+ tree index?") == "procedural"


def test_heuristic_planner_maps_to_expected_configs():
    planner = HeuristicQueryPlanner(
        RAGConfig(
            top_k=5,
            num_candidates=20,
            rerank_mode="cross_encoder",
            planner_mode="heuristic",
        )
    )

    definition_cfg = planner.plan("What is a foreign key?")
    assert planner.last_decision.category == "definitional"
    assert definition_cfg.use_hyde is False
    assert definition_cfg.rerank_mode == "none"
    assert definition_cfg.ranker_weights["bm25"] > definition_cfg.ranker_weights["faiss"]

    analytical_cfg = planner.plan("Compare OLTP and analytics workloads in terms of latency and schema design.")
    assert planner.last_decision.category == "analytical"
    assert analytical_cfg.use_hyde is True
    assert analytical_cfg.rerank_mode == "cross_encoder"
    assert analytical_cfg.num_candidates >= 40


def test_resolve_query_config_returns_planner_metadata():
    cfg = RAGConfig(planner_mode="heuristic")
    planner = build_query_planner(cfg)

    effective_cfg, planner_info = resolve_query_config(
        "Compare OLTP and OLAP systems.",
        cfg,
        planner,
    )

    assert planner_info["planner_enabled"] is True
    assert planner_info["query_category"] == "analytical"
    assert planner_info["planner_config_diff"]
    assert effective_cfg.use_hyde is True


def test_build_query_planner_supports_cost_based_and_adaptive_modes():
    assert isinstance(build_query_planner(RAGConfig(planner_mode="cost_based")), CostBasedQueryPlanner)
    assert isinstance(build_query_planner(RAGConfig(planner_mode="adaptive")), AdaptiveQueryPlanner)


def test_cost_based_planner_selects_pareto_plan_within_budget():
    cfg = RAGConfig(
        planner_mode="cost_based",
        latency_budget_ms=950,
        top_k=5,
        num_candidates=30,
        rerank_mode="none",
    )
    planner = build_query_planner(cfg)

    effective_cfg, planner_info = resolve_query_config(
        "Compare OLTP and OLAP workloads in terms of latency and throughput.",
        cfg,
        planner,
    )

    assert planner_info["planner_enabled"] is True
    assert planner_info["selected_plan_id"]
    assert planner_info["estimated_latency_ms"] is not None
    assert planner_info["estimated_quality"] is not None
    assert planner_info["pareto_frontier"]
    assert planner_info["planner_used_cache"] is False
    assert planner_info["query_category"] == "analytical"
    assert effective_cfg.top_k >= cfg.top_k
    assert effective_cfg.num_candidates >= effective_cfg.top_k
    assert any(candidate["within_budget"] for candidate in planner_info["pareto_frontier"])


def test_adaptive_planner_reuses_cached_plan_after_positive_feedback():
    cfg = RAGConfig(
        planner_mode="adaptive",
        latency_budget_ms=900,
        top_k=5,
        num_candidates=30,
        planner_cache_quality_threshold=0.7,
    )
    planner = build_query_planner(cfg)

    first_query = "Compare OLTP and OLAP workloads."
    _, first_info = resolve_query_config(first_query, cfg, planner)
    planner.record_feedback(query=first_query, latency_ms=420.0, quality_signal=0.93)

    second_query = "Compare B+ trees and binary search trees."
    _, second_info = resolve_query_config(second_query, cfg, planner)

    assert first_info["selected_plan_id"] == second_info["selected_plan_id"]
    assert second_info["planner_used_cache"] is True


def test_adaptive_planner_feedback_updates_snapshot():
    cfg = RAGConfig(
        planner_mode="adaptive",
        latency_budget_ms=800,
        top_k=5,
        num_candidates=30,
    )
    planner = build_query_planner(cfg)
    query = "How do I build a B+ tree index?"

    _, planner_info = resolve_query_config(query, cfg, planner)
    planner.record_feedback(query=query, latency_ms=510.0, quality_signal=0.81)
    snapshot = planner.snapshot()
    feedback_key = f"{planner_info['query_category']}:{planner_info['selected_plan_id']}"

    assert feedback_key in snapshot["feedback"]
    assert snapshot["feedback"][feedback_key]["observations"] == 1
    assert snapshot["feedback"][feedback_key]["quality_ema"] == pytest.approx(0.81)


@patch("src.main.answer", return_value=iter(["planned answer"]))
@patch("src.main.retrieve_and_rank_chunks", return_value=(["context"], [0], [0.9], None, None))
def test_get_answer_uses_planned_config(mock_retrieve, mock_answer):
    cfg = RAGConfig(
        top_k=5,
        num_candidates=20,
        planner_mode="heuristic",
        rerank_mode="cross_encoder",
    )
    planner = build_query_planner(cfg)
    args = argparse.Namespace(system_prompt_mode="baseline", double_prompt=False)
    logger = MagicMock()
    console = MagicMock()
    artifacts = {
        "chunks": ["context"],
        "sources": ["doc"],
        "meta": [{"page_numbers": [1]}],
        "planner": planner,
    }

    answer_text, _, _ = get_answer(
        question="What is a foreign key?",
        cfg=cfg,
        args=args,
        logger=logger,
        console=console,
        artifacts=artifacts,
        is_test_mode=True,
    )

    planned_cfg = mock_retrieve.call_args.kwargs["cfg"]
    assert planned_cfg.rerank_mode == "none"
    assert planned_cfg.use_hyde is False
    assert answer_text == "planned answer"
