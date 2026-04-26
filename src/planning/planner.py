from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from copy import deepcopy

from src.config import RAGConfig


@dataclass
class PlanDecision:
    planner_name: str
    query: str
    category: str
    features: Dict[str, Any] = field(default_factory=dict)
    rationale: List[str] = field(default_factory=list)
    config_diff: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryPlanner(ABC):
    """
    Abstract base for query planners.
    """

    def __init__(self, base_cfg: RAGConfig):
        self.base_cfg = deepcopy(base_cfg)
        self.last_decision: Optional[PlanDecision] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of the planner (for logging)."""

    @abstractmethod
    def plan(self, query: str) -> RAGConfig:
        """
        Subclasses must override this to return an updated RAGConfig.
        """

    def record_feedback(
        self,
        *,
        query: str,
        latency_ms: Optional[float] = None,
        quality_signal: Optional[float] = None,
    ) -> None:
        """Optional online update hook for adaptive planners."""
        return None

    # ---- helper for subclasses ----
    def _config_diff(self, new_cfg: RAGConfig) -> Dict[str, Dict[str, Any]]:
        base_dict = self.base_cfg.to_dict()
        new_dict = new_cfg.to_dict()
        diff: Dict[str, Dict[str, Any]] = {}
        for key in sorted(set(base_dict) | set(new_dict)):
            if base_dict.get(key) != new_dict.get(key):
                diff[key] = {"before": base_dict.get(key), "after": new_dict.get(key)}
        return diff

    def _record_decision(
        self,
        *,
        query: str,
        category: str,
        features: Optional[Dict[str, Any]],
        rationale: Optional[List[str]],
        new_cfg: RAGConfig,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.last_decision = PlanDecision(
            planner_name=self.name,
            query=query,
            category=category,
            features=features or {},
            rationale=rationale or [],
            config_diff=self._config_diff(new_cfg),
            metadata=metadata or {},
        )
        self._log_decision(new_cfg)

    def _log_decision(self, new_cfg: RAGConfig) -> None:
        return None
