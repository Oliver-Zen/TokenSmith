from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.config import RAGConfig
from src.planning.heuristics import HeuristicQueryPlanner
from src.planning.planner import QueryPlanner


@dataclass(frozen=True)
class PlanSpec:
    plan_id: str
    category: str
    ensemble_method: str
    ranker_weights: Dict[str, float]
    candidate_factor: int
    top_k_delta: int = 0
    use_hyde: bool = False
    rerank_mode: str = "none"
    use_double_prompt: bool = False
    quality_bias: float = 0.0
    latency_bias_ms: float = 0.0


@dataclass
class PlanEstimate:
    spec: PlanSpec
    cfg: RAGConfig
    predicted_latency_ms: float
    predicted_quality: float
    within_budget: bool
    utility: float


@dataclass
class PlanFeedback:
    quality_ema: float = 0.0
    latency_ema: float = 0.0
    observations: int = 0

    def update(self, quality: Optional[float], latency_ms: Optional[float], alpha: float) -> None:
        if quality is not None:
            self.quality_ema = quality if self.observations == 0 else (alpha * quality + (1 - alpha) * self.quality_ema)
        if latency_ms is not None:
            self.latency_ema = latency_ms if self.observations == 0 else (alpha * latency_ms + (1 - alpha) * self.latency_ema)
        self.observations += 1


@dataclass
class CacheEntry:
    plan_id: str
    quality_ema: float
    observations: int


class CostBasedQueryPlanner(QueryPlanner):
    def __init__(self, base_cfg: RAGConfig, adaptive: bool = False):
        super().__init__(base_cfg)
        self.adaptive = adaptive
        self.classifier = HeuristicQueryPlanner(base_cfg)
        self._feedback: Dict[Tuple[str, str], PlanFeedback] = {}
        self._cache: "OrderedDict[str, CacheEntry]" = OrderedDict()

    @property
    def name(self) -> str:
        return "adaptive" if self.adaptive else "cost_based"

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights.values()) or 1.0
        return {name: value / total for name, value in weights.items()}

    def _category_profiles(self, category: str) -> Dict[str, Iterable[Any]]:
        include_index = self.base_cfg.ranker_weights.get("index_keywords", 0.0) > 0.0
        hybrid_weights = {"faiss": 0.55, "bm25": 0.45}
        if include_index:
            hybrid_weights["index_keywords"] = 0.10
            hybrid_weights["faiss"] = 0.50
            hybrid_weights["bm25"] = 0.40

        if category == "definitional":
            return {
                "profiles": [
                    ("lexical", "linear", {"faiss": 0.2, "bm25": 0.8}, 0.08),
                    ("hybrid", "rrf", {"faiss": 0.45, "bm25": 0.55}, 0.03),
                    ("semantic", "rrf", {"faiss": 0.7, "bm25": 0.3}, -0.02),
                ],
                "candidate_factors": [3, 4, 5],
                "top_k_deltas": [0],
                "hyde_options": [False],
                "rerank_options": ["none", "cross_encoder"],
                "double_prompt_options": [False],
            }

        if category == "explanatory":
            return {
                "profiles": [
                    ("balanced", "rrf", {"faiss": 0.6, "bm25": 0.4}, 0.05),
                    ("hybrid", "rrf", hybrid_weights, 0.08),
                    ("lexical", "linear", {"faiss": 0.35, "bm25": 0.65}, 0.0),
                ],
                "candidate_factors": [4, 5, 6],
                "top_k_deltas": [0, 1],
                "hyde_options": [False, True],
                "rerank_options": ["none", "cross_encoder"],
                "double_prompt_options": [False, True],
            }

        if category == "analytical":
            return {
                "profiles": [
                    ("hybrid", "rrf", hybrid_weights, 0.08),
                    ("semantic", "rrf", {"faiss": 0.7, "bm25": 0.3}, 0.02),
                    ("index_hybrid", "rrf", {**hybrid_weights, "index_keywords": 0.15} if include_index else hybrid_weights, 0.10),
                ],
                "candidate_factors": [6, 8, 10],
                "top_k_deltas": [0, 2],
                "hyde_options": [False, True],
                "rerank_options": ["none", "cross_encoder"],
                "double_prompt_options": [False, True],
            }

        return {
            "profiles": [
                ("balanced", "rrf", hybrid_weights, 0.07),
                ("lexical", "linear", {"faiss": 0.4, "bm25": 0.6}, 0.03),
                ("semantic", "rrf", {"faiss": 0.65, "bm25": 0.35}, 0.02),
            ],
            "candidate_factors": [5, 6, 8],
            "top_k_deltas": [0, 1],
            "hyde_options": [False, True],
            "rerank_options": ["none", "cross_encoder"],
            "double_prompt_options": [False, True],
        }

    def enumerate_plans(self, category: str) -> List[PlanSpec]:
        profiles = self._category_profiles(category)
        plans: List[PlanSpec] = []
        seen: set = set()

        for profile_name, ensemble_method, weights, quality_bias in profiles["profiles"]:
            normalized_weights = self._normalize_weights(dict(weights))
            for candidate_factor in profiles["candidate_factors"]:
                for top_k_delta in profiles["top_k_deltas"]:
                    for use_hyde in profiles["hyde_options"]:
                        for rerank_mode in profiles["rerank_options"]:
                            for use_double_prompt in profiles["double_prompt_options"]:
                                if category == "definitional" and use_hyde:
                                    continue
                                if category == "definitional" and use_double_prompt:
                                    continue
                                if category in {"definitional", "explanatory"} and use_double_prompt and not use_hyde:
                                    continue
                                if use_double_prompt and rerank_mode == "none" and category != "analytical":
                                    continue
                                if category == "definitional" and rerank_mode == "cross_encoder" and candidate_factor <= 3:
                                    continue

                                signature = (
                                    profile_name,
                                    ensemble_method,
                                    tuple(sorted(normalized_weights.items())),
                                    candidate_factor,
                                    top_k_delta,
                                    use_hyde,
                                    rerank_mode,
                                    use_double_prompt,
                                )
                                if signature in seen:
                                    continue
                                seen.add(signature)

                                plan_id = (
                                    f"{category[:4]}_{profile_name}_c{candidate_factor}"
                                    f"_k{top_k_delta:+d}_{'hyde' if use_hyde else 'plain'}"
                                    f"_{'rerank' if rerank_mode == 'cross_encoder' else 'base'}"
                                    f"_{'double' if use_double_prompt else 'single'}"
                                )
                                plans.append(
                                    PlanSpec(
                                        plan_id=plan_id,
                                        category=category,
                                        ensemble_method=ensemble_method,
                                        ranker_weights=normalized_weights,
                                        candidate_factor=candidate_factor,
                                        top_k_delta=top_k_delta,
                                        use_hyde=use_hyde,
                                        rerank_mode=rerank_mode,
                                        use_double_prompt=use_double_prompt,
                                        quality_bias=quality_bias,
                                        latency_bias_ms=20.0 if ensemble_method == "rrf" else 0.0,
                                    )
                                )
        return plans

    def _cache_key(self, category: str, features: Dict[str, Any]) -> str:
        comparison_bucket = min(int(features.get("comparison_hits", 0)), 2)
        procedural_bucket = min(int(features.get("procedural_hits", 0)), 2)
        length_bucket = "short" if features.get("is_short_query") else "long" if features.get("is_long_query") else "mid"
        return "|".join(
            [
                category,
                str(features.get("question_word", "other")),
                length_bucket,
                f"cmp{comparison_bucket}",
                f"proc{procedural_bucket}",
                "multi" if features.get("has_multi_clause") else "single",
            ]
        )

    def _cfg_for_spec(self, spec: PlanSpec) -> RAGConfig:
        cfg = deepcopy(self.base_cfg)
        top_k = max(3, cfg.top_k + spec.top_k_delta)
        num_candidates = max(top_k, top_k * spec.candidate_factor)
        rerank_top_k = min(max(cfg.rerank_top_k, top_k), num_candidates)
        return cfg.with_updates(
            top_k=top_k,
            num_candidates=num_candidates,
            ensemble_method=spec.ensemble_method,
            ranker_weights=spec.ranker_weights,
            use_hyde=spec.use_hyde,
            rerank_mode=spec.rerank_mode,
            rerank_top_k=rerank_top_k if spec.rerank_mode == "cross_encoder" else top_k,
            use_double_prompt=spec.use_double_prompt,
        )

    def _feedback_for(self, category: str, plan_id: str) -> PlanFeedback:
        return self._feedback.get((category, plan_id), PlanFeedback())

    def _estimate_latency_ms(self, cfg: RAGConfig, category: str, features: Dict[str, Any], spec: PlanSpec) -> float:
        token_count = int(features.get("token_count", 0))
        comparison_hits = int(features.get("comparison_hits", 0))
        explanation_hits = int(features.get("explanation_hits", 0))
        procedural_hits = int(features.get("procedural_hits", 0))

        latency = 85.0
        latency += cfg.num_candidates * 4.0
        latency += cfg.top_k * 14.0
        if cfg.ensemble_method == "rrf":
            latency += 25.0
        if cfg.use_hyde:
            latency += 280.0
        if cfg.rerank_mode == "cross_encoder":
            latency += 48.0 * min(cfg.rerank_top_k, cfg.top_k)
        if cfg.use_double_prompt:
            latency += 240.0

        complexity = 1.0
        complexity += min(token_count, 24) * 0.015
        complexity += comparison_hits * 0.08
        complexity += explanation_hits * 0.04
        complexity += procedural_hits * 0.05
        if category == "analytical":
            complexity += 0.12
        elif category == "procedural":
            complexity += 0.08

        latency = latency * complexity + spec.latency_bias_ms
        feedback = self._feedback_for(category, spec.plan_id)
        if feedback.observations and feedback.latency_ema > 0:
            blend = min(0.65, 0.15 + 0.1 * feedback.observations)
            latency = (1 - blend) * latency + blend * feedback.latency_ema
        return round(latency, 2)

    def _estimate_quality(self, cfg: RAGConfig, category: str, features: Dict[str, Any], spec: PlanSpec) -> float:
        token_count = int(features.get("token_count", 0))
        comparison_hits = int(features.get("comparison_hits", 0))
        explanation_hits = int(features.get("explanation_hits", 0))
        procedural_hits = int(features.get("procedural_hits", 0))
        entity_density = float(features.get("entity_density", 0.0))

        category_base = {
            "definitional": 0.60,
            "explanatory": 0.64,
            "analytical": 0.67,
            "procedural": 0.65,
        }[category]
        quality = category_base + spec.quality_bias

        bm25_weight = cfg.ranker_weights.get("bm25", 0.0)
        faiss_weight = cfg.ranker_weights.get("faiss", 0.0)
        keyword_weight = cfg.ranker_weights.get("index_keywords", 0.0)

        if category == "definitional":
            quality += bm25_weight * 0.14 + entity_density * 0.05
            quality -= 0.03 if cfg.use_hyde else 0.0
        elif category == "explanatory":
            quality += faiss_weight * 0.06 + bm25_weight * 0.05 + explanation_hits * 0.01
            quality += 0.03 if cfg.use_hyde else 0.0
        elif category == "analytical":
            quality += faiss_weight * 0.07 + bm25_weight * 0.06 + keyword_weight * 0.03
            quality += comparison_hits * 0.02
            quality += 0.07 if cfg.use_hyde else -0.01
        else:
            quality += bm25_weight * 0.06 + faiss_weight * 0.04 + procedural_hits * 0.02
            quality += 0.05 if cfg.use_hyde else 0.0

        quality += min(cfg.num_candidates / max(cfg.top_k, 1), 8) * 0.01
        quality += min(token_count, 20) * 0.002
        quality += 0.05 if cfg.rerank_mode == "cross_encoder" and category in {"analytical", "procedural"} else 0.0
        quality += 0.02 if cfg.rerank_mode == "cross_encoder" and category == "explanatory" else 0.0
        quality += 0.04 if cfg.use_double_prompt and category in {"analytical", "explanatory"} else 0.0
        quality -= 0.02 if cfg.use_double_prompt and category == "definitional" else 0.0

        feedback = self._feedback_for(category, spec.plan_id)
        if feedback.observations:
            blend = min(0.60, 0.15 + 0.1 * feedback.observations)
            quality = (1 - blend) * quality + blend * feedback.quality_ema

        return round(max(0.05, min(0.99, quality)), 4)

    def evaluate_plan(self, query: str, category: str, features: Dict[str, Any], spec: PlanSpec) -> PlanEstimate:
        cfg = self._cfg_for_spec(spec)
        latency_budget = getattr(self.base_cfg, "latency_budget_ms", 900)
        predicted_latency = self._estimate_latency_ms(cfg, category, features, spec)
        predicted_quality = self._estimate_quality(cfg, category, features, spec)
        within_budget = predicted_latency <= latency_budget
        overflow_penalty = max(0.0, predicted_latency - latency_budget) / max(latency_budget, 1)
        utility = predicted_quality - (0.18 * predicted_latency / max(latency_budget, 1)) - (0.15 * overflow_penalty)
        return PlanEstimate(
            spec=spec,
            cfg=cfg,
            predicted_latency_ms=predicted_latency,
            predicted_quality=predicted_quality,
            within_budget=within_budget,
            utility=round(utility, 4),
        )

    def estimate_frontier(self, evaluations: List[PlanEstimate]) -> List[PlanEstimate]:
        frontier: List[PlanEstimate] = []
        for candidate in sorted(evaluations, key=lambda item: (item.predicted_latency_ms, -item.predicted_quality)):
            dominated = False
            for other in evaluations:
                if other is candidate:
                    continue
                if (
                    other.predicted_latency_ms <= candidate.predicted_latency_ms
                    and other.predicted_quality >= candidate.predicted_quality
                    and (
                        other.predicted_latency_ms < candidate.predicted_latency_ms
                        or other.predicted_quality > candidate.predicted_quality
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                frontier.append(candidate)
        return frontier

    def _estimate_from_cached_plan(
        self,
        *,
        query: str,
        category: str,
        features: Dict[str, Any],
        cached_plan_id: str,
    ) -> Optional[Tuple[RAGConfig, Dict[str, Any], List[str]]]:
        for spec in self.enumerate_plans(category):
            if spec.plan_id != cached_plan_id:
                continue
            estimate = self.evaluate_plan(query, category, features, spec)
            rationale = ["reused cached plan for a matching query signature"]
            metadata = {
                "selected_plan_id": spec.plan_id,
                "planner_mode": self.name,
                "estimated_latency_ms": estimate.predicted_latency_ms,
                "estimated_quality": estimate.predicted_quality,
                "latency_budget_ms": getattr(self.base_cfg, "latency_budget_ms", 900),
                "candidate_count": 1,
                "pareto_frontier": [
                    {
                        "plan_id": spec.plan_id,
                        "predicted_latency_ms": estimate.predicted_latency_ms,
                        "predicted_quality": estimate.predicted_quality,
                        "within_budget": estimate.within_budget,
                    }
                ],
                "used_cache": True,
                "cache_key": self._cache_key(category, features),
                "feedback_observations": self._feedback_for(category, spec.plan_id).observations,
            }
            return estimate.cfg, metadata, rationale
        return None

    def plan(self, query: str) -> RAGConfig:
        features = self.classifier._extract_features(query)
        scores, rationale = self.classifier._score_categories(features)
        priority = {"analytical": 0, "procedural": 1, "explanatory": 2, "definitional": 3}
        category = max(scores.items(), key=lambda item: (item[1], -priority[item[0]]))[0]
        features = {
            **features,
            "category_scores": {name: round(value, 3) for name, value in scores.items()},
        }
        cache_key = self._cache_key(category, features)

        if self.adaptive:
            cached = self._cache.get(cache_key)
            if cached and cached.quality_ema >= self.base_cfg.planner_cache_quality_threshold:
                cached_estimate = self._estimate_from_cached_plan(
                    query=query,
                    category=category,
                    features=features,
                    cached_plan_id=cached.plan_id,
                )
                if cached_estimate is not None:
                    cfg, metadata, cache_rationale = cached_estimate
                    self._cache.move_to_end(cache_key)
                    self._record_decision(
                        query=query,
                        category=category,
                        features=features,
                        rationale=rationale + cache_rationale,
                        new_cfg=cfg,
                        metadata=metadata,
                    )
                    return cfg

        evaluations = [
            self.evaluate_plan(query, category, features, spec)
            for spec in self.enumerate_plans(category)
        ]
        frontier = self.estimate_frontier(evaluations)
        eligible = [candidate for candidate in frontier if candidate.within_budget]
        selected = max(
            eligible or frontier,
            key=lambda item: (
                item.predicted_quality,
                -item.predicted_latency_ms,
                item.utility,
            ),
        )

        rationale = list(rationale)
        if eligible:
            rationale.append("selected the highest-quality Pareto plan within the latency budget")
        else:
            rationale.append("no Pareto plan fit the latency budget, so the best quality/latency tradeoff was chosen")

        metadata = {
            "selected_plan_id": selected.spec.plan_id,
            "planner_mode": self.name,
            "estimated_latency_ms": selected.predicted_latency_ms,
            "estimated_quality": selected.predicted_quality,
            "latency_budget_ms": getattr(self.base_cfg, "latency_budget_ms", 900),
            "candidate_count": len(evaluations),
            "pareto_frontier": [
                {
                    "plan_id": candidate.spec.plan_id,
                    "predicted_latency_ms": candidate.predicted_latency_ms,
                    "predicted_quality": candidate.predicted_quality,
                    "within_budget": candidate.within_budget,
                }
                for candidate in sorted(frontier, key=lambda item: (item.predicted_latency_ms, -item.predicted_quality))
            ],
            "used_cache": False,
            "cache_key": cache_key,
            "feedback_observations": self._feedback_for(category, selected.spec.plan_id).observations,
        }
        self._record_decision(
            query=query,
            category=category,
            features=features,
            rationale=rationale,
            new_cfg=selected.cfg,
            metadata=metadata,
        )
        return selected.cfg

    def record_feedback(
        self,
        *,
        query: str,
        latency_ms: Optional[float] = None,
        quality_signal: Optional[float] = None,
    ) -> None:
        decision = self.last_decision
        if decision is None or decision.query != query:
            return

        plan_id = decision.metadata.get("selected_plan_id")
        if not plan_id:
            return

        category = decision.category
        alpha = self.base_cfg.planner_feedback_alpha
        feedback_key = (category, plan_id)
        feedback = self._feedback.setdefault(feedback_key, PlanFeedback())
        feedback.update(quality_signal, latency_ms, alpha)

        if not self.adaptive:
            return

        cache_key = decision.metadata.get("cache_key") or self._cache_key(category, decision.features)
        quality_ema = feedback.quality_ema if feedback.observations else 0.0
        if quality_ema < self.base_cfg.planner_cache_quality_threshold:
            return

        self._cache[cache_key] = CacheEntry(
            plan_id=plan_id,
            quality_ema=quality_ema,
            observations=feedback.observations,
        )
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self.base_cfg.planner_cache_max_entries:
            self._cache.popitem(last=False)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "feedback": {
                f"{category}:{plan_id}": {
                    "quality_ema": round(state.quality_ema, 4),
                    "latency_ema": round(state.latency_ema, 2),
                    "observations": state.observations,
                }
                for (category, plan_id), state in sorted(self._feedback.items())
            },
            "cache": {
                key: {
                    "plan_id": entry.plan_id,
                    "quality_ema": round(entry.quality_ema, 4),
                    "observations": entry.observations,
                }
                for key, entry in self._cache.items()
            },
        }


class AdaptiveQueryPlanner(CostBasedQueryPlanner):
    def __init__(self, base_cfg: RAGConfig):
        super().__init__(base_cfg, adaptive=True)
