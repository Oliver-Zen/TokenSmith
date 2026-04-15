from __future__ import annotations

import re
from copy import deepcopy
from typing import Dict, List, Tuple

from src.config import RAGConfig
from src.planning.planner import QueryPlanner


QUESTION_WORDS = {"what", "why", "how", "when", "where", "which", "who", "compare", "contrast"}
DEFINITION_PHRASES = {"what is", "what are", "define", "definition", "meaning of", "stands for"}
EXPLANATORY_TERMS = {"why", "explain", "description", "describe", "reason", "intuition"}
ANALYTICAL_TERMS = {
    "compare",
    "contrast",
    "difference",
    "different",
    "versus",
    "vs",
    "tradeoff",
    "trade-off",
    "advantages",
    "disadvantages",
    "better",
    "worse",
    "relationship",
}
PROCEDURAL_TERMS = {"how", "steps", "procedure", "process", "algorithm", "implement", "build", "configure"}


class HeuristicQueryPlanner(QueryPlanner):
    @property
    def name(self) -> str:
        return "heuristic"

    def __init__(self, base_cfg: RAGConfig):
        super().__init__(base_cfg)
        self.base_cfg = deepcopy(base_cfg)

    def _tokenize(self, query: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9][A-Za-z0-9_+-]*", query)

    def _extract_features(self, query: str) -> Dict[str, object]:
        tokens = self._tokenize(query)
        lower_tokens = [token.lower() for token in tokens]
        query_lower = query.lower().strip()
        first_token = lower_tokens[0] if lower_tokens else ""

        capitalized = [
            token for idx, token in enumerate(tokens)
            if token[:1].isupper() and not token.isupper() and idx > 0
        ]
        entity_density = len(capitalized) / max(len(tokens), 1)
        comparison_hits = sum(1 for token in lower_tokens if token in ANALYTICAL_TERMS)
        explanation_hits = sum(1 for token in lower_tokens if token in EXPLANATORY_TERMS)
        procedural_hits = sum(1 for token in lower_tokens if token in PROCEDURAL_TERMS)
        definition_hits = sum(1 for phrase in DEFINITION_PHRASES if phrase in query_lower)
        has_multi_clause = any(marker in query_lower for marker in [",", ";", " and ", " or ", " versus ", " vs "])

        return {
            "token_count": len(tokens),
            "question_word": first_token if first_token in QUESTION_WORDS else "other",
            "entity_density": round(entity_density, 3),
            "comparison_hits": comparison_hits,
            "explanation_hits": explanation_hits,
            "procedural_hits": procedural_hits,
            "definition_hits": definition_hits,
            "has_multi_clause": has_multi_clause,
            "is_short_query": len(tokens) <= 8,
            "is_long_query": len(tokens) >= 14,
        }

    def _score_categories(self, features: Dict[str, object]) -> Tuple[Dict[str, float], List[str]]:
        token_count = int(features["token_count"])
        question_word = str(features["question_word"])
        entity_density = float(features["entity_density"])
        comparison_hits = int(features["comparison_hits"])
        explanation_hits = int(features["explanation_hits"])
        procedural_hits = int(features["procedural_hits"])
        definition_hits = int(features["definition_hits"])
        has_multi_clause = bool(features["has_multi_clause"])
        is_short_query = bool(features["is_short_query"])
        is_long_query = bool(features["is_long_query"])

        scores = {
            "definitional": 0.0,
            "explanatory": 0.0,
            "analytical": 0.0,
            "procedural": 0.0,
        }
        rationale: List[str] = []

        if definition_hits:
            scores["definitional"] += 4.0 + definition_hits
            rationale.append("matched definition-oriented phrasing")
        if question_word == "what":
            scores["definitional"] += 1.5
            scores["explanatory"] += 0.5
        if question_word == "why":
            scores["explanatory"] += 3.0
            rationale.append("question starts with 'why'")
        if question_word == "how":
            scores["procedural"] += 3.0
            rationale.append("question starts with 'how'")
        if question_word in {"compare", "contrast"}:
            scores["analytical"] += 4.0
            rationale.append("question explicitly asks for comparison")

        scores["analytical"] += comparison_hits * 2.5
        scores["explanatory"] += explanation_hits * 2.0
        scores["procedural"] += procedural_hits * 1.8

        if is_short_query:
            scores["definitional"] += 1.2
        if is_long_query:
            scores["analytical"] += 1.5
            scores["explanatory"] += 1.0
        if has_multi_clause:
            scores["analytical"] += 1.2
            scores["procedural"] += 0.6
        if entity_density >= 0.2:
            scores["definitional"] += 0.8
            rationale.append("high entity density suggests a focused concept lookup")
        if token_count >= 18:
            scores["procedural"] += 0.8

        if comparison_hits:
            rationale.append("contains comparison terminology")
        if procedural_hits >= 2:
            rationale.append("contains multiple procedural cues")
        if explanation_hits:
            rationale.append("contains explanation-oriented language")

        return scores, rationale

    def classify(self, query: str) -> str:
        features = self._extract_features(query)
        scores, _ = self._score_categories(features)
        priority = {"analytical": 0, "procedural": 1, "explanatory": 2, "definitional": 3}
        return max(scores.items(), key=lambda item: (item[1], -priority[item[0]]))[0]

    def _build_weights(self, *, faiss: float, bm25: float, index_keywords: float = 0.0) -> Dict[str, float]:
        weights: Dict[str, float] = {"faiss": faiss, "bm25": bm25}
        include_index = (
            self.base_cfg.ranker_weights.get("index_keywords", 0.0) > 0.0
            or index_keywords > 0.0
        )
        if include_index:
            weights["index_keywords"] = index_keywords

        total = sum(weights.values()) or 1.0
        return {name: value / total for name, value in weights.items()}

    def _config_for_category(self, category: str) -> RAGConfig:
        cfg = deepcopy(self.base_cfg)

        if category == "definitional":
            return cfg.with_updates(
                ensemble_method="linear",
                ranker_weights=self._build_weights(faiss=0.2, bm25=0.8, index_keywords=0.0),
                num_candidates=max(cfg.top_k * 3, min(cfg.num_candidates, cfg.top_k * 4)),
                use_hyde=False,
                rerank_mode="none",
                rerank_top_k=cfg.top_k,
                use_double_prompt=False,
            )

        if category == "explanatory":
            return cfg.with_updates(
                ensemble_method="rrf",
                ranker_weights=self._build_weights(faiss=0.65, bm25=0.35, index_keywords=0.0),
                num_candidates=max(cfg.num_candidates, cfg.top_k * 4),
                use_hyde=False,
                rerank_mode="none",
                rerank_top_k=max(cfg.rerank_top_k, min(cfg.top_k, 6)),
                use_double_prompt=False,
            )

        if category == "analytical":
            return cfg.with_updates(
                ensemble_method="rrf",
                ranker_weights=self._build_weights(faiss=0.55, bm25=0.35, index_keywords=0.10),
                num_candidates=max(cfg.num_candidates, cfg.top_k * 8),
                use_hyde=True,
                rerank_mode="cross_encoder",
                rerank_top_k=max(cfg.rerank_top_k, min(cfg.top_k, 8)),
                use_double_prompt=False,
            )

        return cfg.with_updates(
            ensemble_method="rrf",
            ranker_weights=self._build_weights(faiss=0.45, bm25=0.45, index_keywords=0.10),
            num_candidates=max(cfg.num_candidates, cfg.top_k * 6),
            use_hyde=False,
            rerank_mode="cross_encoder",
            rerank_top_k=max(cfg.rerank_top_k, min(cfg.top_k, 8)),
            use_double_prompt=False,
        )

    def plan(self, query: str) -> RAGConfig:
        features = self._extract_features(query)
        scores, rationale = self._score_categories(features)
        priority = {"analytical": 0, "procedural": 1, "explanatory": 2, "definitional": 3}
        category = max(scores.items(), key=lambda item: (item[1], -priority[item[0]]))[0]
        cfg = self._config_for_category(category)

        features = {
            **features,
            "category_scores": {name: round(value, 3) for name, value in scores.items()},
        }
        self._record_decision(
            query=query,
            category=category,
            features=features,
            rationale=rationale,
            new_cfg=cfg,
        )
        return cfg
