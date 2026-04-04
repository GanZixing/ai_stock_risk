from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


# Reserved factors are intentionally inactive placeholders.
# They are globally registered to guarantee schema/config/output presence.
RESERVED_FACTORS = ["AAA", "BBB", "CCC"]


# Global factor registry for the full platform.
# Future factors should be added here and then configured through model config only.
GLOBAL_FACTOR_REGISTRY: Dict[str, List[str]] = {
    "statistical": [
        "annualized_volatility",
        "downside_volatility",
        "beta",
        "maximum_drawdown",
        "historical_var_95",
        "historical_cvar_95",
        "liquidity_risk_proxy",
        *RESERVED_FACTORS,
    ],
    "structural": [
        "profitability_risk",
        "leverage_risk",
        "cash_flow_quality_risk",
        "valuation_stress_risk",
        "growth_deceleration_risk",
        "size_concentration_risk",
        "sector_policy_risk",
        "altman_z_score",
        *RESERVED_FACTORS,
    ],
}


@dataclass
class FactorConfig:
    weight: float
    min_value: float
    max_value: float
    direction: str = "higher_worse"


@dataclass
class ModelConfig:
    factor_thresholds: Dict[str, FactorConfig] = field(default_factory=dict)


@dataclass
class StatisticalRiskInput:
    ticker: str
    benchmark_ticker: str = "SPY"
    lookback_window: int = 252
    price_history: Optional[Sequence[Mapping[str, Any]]] = None
    benchmark_history: Optional[Sequence[Mapping[str, Any]]] = None
    # Reserved placeholders are included in input schema for forward compatibility.
    reserved_factors: Dict[str, Optional[float]] = field(default_factory=lambda: default_reserved_factor_values())


@dataclass
class StructuralRiskInput:
    ticker: str
    financial_data: Optional[Mapping[str, Any]] = None
    company_profile: Optional[Mapping[str, Any]] = None
    industry_metadata: Optional[Mapping[str, Any]] = None
    # Reserved placeholders are included in input schema for forward compatibility.
    reserved_factors: Dict[str, Optional[float]] = field(default_factory=lambda: default_reserved_factor_values())


@dataclass
class FactorResult:
    factor: str
    raw_value: Optional[float]
    normalized_score: float
    weight: float
    thresholds: Dict[str, float]


@dataclass
class ModelResult:
    model_name: str
    overall_score: float
    risk_level: str
    factor_breakdown: List[FactorResult]
    raw_metrics: Dict[str, Optional[float]]
    weights_used: Dict[str, float]
    # Reserved factor status is emitted for transparency.
    reserved_factors: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "overall_score": float(self.overall_score),
            "risk_level": self.risk_level,
            "factor_breakdown": [asdict(f) for f in self.factor_breakdown],
            "raw_metrics": self.raw_metrics,
            "weights_used": self.weights_used,
            "reserved_factors": self.reserved_factors,
        }


class BaseFactor(ABC):
    @abstractmethod
    def compute(self, data: Mapping[str, Any]) -> Optional[float]:
        raise NotImplementedError


class BaseModel(ABC):
    model_name: str

    @abstractmethod
    def evaluate(self, ticker: str, **kwargs) -> ModelResult:
        raise NotImplementedError


class DataAdapter(ABC):
    @abstractmethod
    def get_data(self, ticker: str, **kwargs) -> Mapping[str, Any]:
        raise NotImplementedError


def get_reserved_factor_config() -> Dict[str, FactorConfig]:
    return {
        factor: FactorConfig(weight=0, min_value=0.0, max_value=1.0, direction="higher_worse")
        for factor in RESERVED_FACTORS
    }


def default_reserved_factor_values(default_value: Optional[float] = None) -> Dict[str, Optional[float]]:
    return {factor: default_value for factor in RESERVED_FACTORS}


def apply_reserved_factor_values(
    raw_metrics: Dict[str, Optional[float]],
    reserved_overrides: Optional[Mapping[str, Optional[float]]] = None,
) -> Dict[str, Optional[float]]:
    """Future extension hook for activating reserved factors without model refactors."""
    merged = dict(raw_metrics)
    overrides = reserved_overrides or {}
    for factor in RESERVED_FACTORS:
        merged[factor] = overrides.get(factor, merged.get(factor))
    return merged


def get_reserved_factor_status(
    raw_metrics: Mapping[str, Optional[float]],
    config: Mapping[str, FactorConfig],
) -> Dict[str, Dict[str, Any]]:
    status: Dict[str, Dict[str, Any]] = {}
    for factor in RESERVED_FACTORS:
        factor_cfg = config.get(factor)
        weight = factor_cfg.weight if factor_cfg is not None else 0.0
        raw_value = raw_metrics.get(factor)
        # Current behavior: reserved factors default to zero normalized score unless activated.
        normalized_score = 0.0 if weight == 0 else 0.0
        status[factor] = {
            "raw_value": raw_value,
            "normalized_score": normalized_score,
            "weight": weight,
            "active": weight > 0,
            "note": "Reserved inactive placeholder factor",
        }
    return status
