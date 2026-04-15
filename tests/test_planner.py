import argparse
from unittest.mock import MagicMock, patch

import pytest

from src.config import RAGConfig
from src.main import build_query_planner, get_answer, resolve_query_config
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
