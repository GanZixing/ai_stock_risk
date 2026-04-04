from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


class StructuralRiskDataAdapter(ABC):
    """Adapter interface for company-level structural risk data."""

    @abstractmethod
    def get_company_data(
        self,
        ticker: str,
    ) -> Mapping[str, Any]:
        """Return a mapping with financials, profile, and optional metadata."""
        raise NotImplementedError


@dataclass
class FactorConfig:
    weight: float
    min_value: float
    max_value: float
    direction: str = "higher_worse"


@dataclass
class StructuralRiskModelConfig:
    factor_thresholds: Dict[str, FactorConfig] = field(default_factory=dict)

    @staticmethod
    def default() -> "StructuralRiskModelConfig":
        return StructuralRiskModelConfig(
            factor_thresholds={
                "profitability_risk": FactorConfig(weight=15, min_value=0.0, max_value=1.0, direction="higher_worse"),
                "leverage_risk": FactorConfig(weight=15, min_value=0.0, max_value=1.0, direction="higher_worse"),
                "cash_flow_quality_risk": FactorConfig(weight=15, min_value=0.0, max_value=1.0, direction="higher_worse"),
                "valuation_stress_risk": FactorConfig(weight=10, min_value=0.0, max_value=1.0, direction="higher_worse"),
                "growth_deceleration_risk": FactorConfig(weight=10, min_value=0.0, max_value=1.0, direction="higher_worse"),
                "size_concentration_risk": FactorConfig(weight=10, min_value=0.0, max_value=1.0, direction="higher_worse"),
                "sector_policy_risk": FactorConfig(weight=10, min_value=0.0, max_value=1.0, direction="higher_worse"),
                "altman_z_score": FactorConfig(weight=5, min_value=0.0, max_value=1.0, direction="higher_worse"),
                "AAA": FactorConfig(weight=0, min_value=0.0, max_value=1.0, direction="higher_worse"),
                "BBB": FactorConfig(weight=0, min_value=0.0, max_value=1.0, direction="higher_worse"),
                "CCC": FactorConfig(weight=0, min_value=0.0, max_value=1.0, direction="higher_worse"),
            }
        )


@dataclass
class FactorResult:
    factor: str
    raw_value: Optional[float]
    normalized_score: float
    weight: float
    thresholds: Dict[str, float]


@dataclass
class StructuralRiskModelResult:
    model_name: str
    overall_score: float
    risk_level: str
    factor_breakdown: List[FactorResult]
    raw_metrics: Dict[str, Optional[float]]
    weights_used: Dict[str, float]
    missing_data_notes: List[str]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StructuralRiskModel:
    model_name = "StructuralRiskModel"

    def __init__(self, config: Optional[StructuralRiskModelConfig] = None):
        self.config = config or StructuralRiskModelConfig.default()

    def evaluate(
        self,
        ticker: str,
        financial_data: Optional[Mapping[str, Any]] = None,
        company_profile: Optional[Mapping[str, Any]] = None,
        industry_metadata: Optional[Mapping[str, Any]] = None,
        data_adapter: Optional[StructuralRiskDataAdapter] = None,
    ) -> StructuralRiskModelResult:
        if data_adapter is not None:
            data = data_adapter.get_company_data(ticker)
            financial_data = financial_data or data.get("financial_data")
            company_profile = company_profile or data.get("company_profile")
            industry_metadata = industry_metadata or data.get("industry_metadata")

        if financial_data is None or company_profile is None:
            raise ValueError("financial_data and company_profile are required inputs.")

        missing_notes: List[str] = []
        raw_metrics = self._calculate_raw_metrics(financial_data, company_profile, industry_metadata, missing_notes)
        factor_breakdown = self._normalize_factors(raw_metrics)
        overall_score = self._aggregate_score(factor_breakdown)
        risk_level = self._risk_level_label(overall_score)

        return StructuralRiskModelResult(
            model_name=self.model_name,
            overall_score=round(overall_score, 2),
            risk_level=risk_level,
            factor_breakdown=factor_breakdown,
            raw_metrics=raw_metrics,
            weights_used={factor: cfg.weight for factor, cfg in self.config.factor_thresholds.items()},
            missing_data_notes=missing_notes,
            explanation=self._explanation(factor_breakdown),
        )

    def _calculate_raw_metrics(
        self,
        financial_data: Mapping[str, Any],
        company_profile: Mapping[str, Any],
        industry_metadata: Optional[Mapping[str, Any]],
        missing_notes: List[str],
    ) -> Dict[str, Optional[float]]:
        income = self._as_dict(financial_data.get("income_statement", {}))
        balance = self._as_dict(financial_data.get("balance_sheet", {}))
        cash_flow = self._as_dict(financial_data.get("cash_flow", {}))
        market = self._as_dict(financial_data.get("market", {}))

        profitability = self._profitability_risk(income, missing_notes)
        leverage = self._leverage_risk(balance, missing_notes)
        cash_quality = self._cash_flow_quality_risk(cash_flow, income, missing_notes)
        valuation = self._valuation_stress_risk(market, missing_notes)
        growth = self._growth_deceleration_risk(income, missing_notes)
        size_concentration = self._size_concentration_risk(market, company_profile, missing_notes)
        sector_policy = self._sector_policy_risk(company_profile, industry_metadata, missing_notes)
        altman = self._altman_z_score_risk(financial_data, missing_notes)

        return {
            "profitability_risk": profitability,
            "leverage_risk": leverage,
            "cash_flow_quality_risk": cash_quality,
            "valuation_stress_risk": valuation,
            "growth_deceleration_risk": growth,
            "size_concentration_risk": size_concentration,
            "sector_policy_risk": sector_policy,
            "altman_z_score": altman,
            "AAA": None,
            "BBB": None,
            "CCC": None,
        }

    def _normalize_factors(self, raw_metrics: Mapping[str, Optional[float]]) -> List[FactorResult]:
        results: List[FactorResult] = []
        for factor, cfg in self.config.factor_thresholds.items():
            raw_value = raw_metrics.get(factor)
            normalized_score = self._normalize_value(raw_value, cfg)
            results.append(
                FactorResult(
                    factor=factor,
                    raw_value=raw_value,
                    normalized_score=normalized_score,
                    weight=cfg.weight,
                    thresholds={
                        "min_value": cfg.min_value,
                        "max_value": cfg.max_value,
                        "direction": cfg.direction,
                    },
                )
            )
        return results

    def _aggregate_score(self, factors: Iterable[FactorResult]) -> float:
        weighted_sum = 0.0
        total_weight = 0.0
        for factor in factors:
            if factor.raw_value is None:
                continue
            weighted_sum += factor.normalized_score * factor.weight
            total_weight += factor.weight
        if total_weight <= 0:
            return 0.0
        return weighted_sum / total_weight

    @staticmethod
    def _risk_level_label(score: float) -> str:
        if score <= 25:
            return "Low"
        if score <= 50:
            return "Moderate"
        if score <= 75:
            return "High"
        return "Extreme"

    @staticmethod
    def _as_dict(data: Any) -> Dict[str, Any]:
        if isinstance(data, Mapping):
            return dict(data)
        return {}

    @staticmethod
    def _get_value(data: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
        for key in keys:
            if key in data and data[key] is not None:
                try:
                    return float(data[key])
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _normalize_ratio(value: Optional[float], minimum: float, maximum: float) -> Optional[float]:
        if value is None or math.isnan(value):
            return None
        if maximum == minimum:
            return None
        return max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))

    @staticmethod
    def _safely_average(values: Sequence[Optional[float]]) -> Optional[float]:
        available = [v for v in values if v is not None]
        if not available:
            return None
        return mean(available)

    def _profitability_risk(self, income: Mapping[str, Any], missing_notes: List[str]) -> Optional[float]:
        net_income = self._get_value(income, ["net_income"])
        gross_margin = self._get_value(income, ["gross_margin"])
        operating_margin = self._get_value(income, ["operating_margin"])
        roe = self._get_value(income, ["roe"])
        roa = self._get_value(income, ["roa"])
        net_income_history = income.get("net_income_history")

        if net_income is None and gross_margin is None and operating_margin is None and roe is None and roa is None:
            missing_notes.append("profitability data is incomplete")
            return None

        sub_scores: List[Optional[float]] = []
        if net_income is not None:
            sub_scores.append(0.0 if net_income >= 0 else 1.0)
        if gross_margin is not None:
            sub_scores.append(1.0 - self._normalize_ratio(gross_margin, 0.0, 0.5) if gross_margin >= 0 else 1.0)
        if operating_margin is not None:
            sub_scores.append(1.0 - self._normalize_ratio(operating_margin, 0.0, 0.3) if operating_margin >= 0 else 1.0)
        if roe is not None:
            sub_scores.append(1.0 - self._normalize_ratio(roe, 0.0, 0.25) if roe >= 0 else 1.0)
        if roa is not None:
            sub_scores.append(1.0 - self._normalize_ratio(roa, 0.0, 0.15) if roa >= 0 else 1.0)
        if isinstance(net_income_history, list) and net_income_history:
            positive_streak = all((x is not None and x >= 0) for x in net_income_history)
            sub_scores.append(0.0 if positive_streak else 1.0)

        return self._safely_average(sub_scores) or 0.0

    def _leverage_risk(self, balance: Mapping[str, Any], missing_notes: List[str]) -> Optional[float]:
        debt = self._get_value(balance, ["total_debt", "debt"])
        equity = self._get_value(balance, ["total_equity", "equity"])
        current_ratio = self._get_value(balance, ["current_ratio"])
        current_assets = self._get_value(balance, ["current_assets"])
        current_liabilities = self._get_value(balance, ["current_liabilities"])
        interest_expense = self._get_value(balance, ["interest_expense"])
        ebit = self._get_value(balance, ["ebit", "operating_income"])

        if debt is None and equity is None and current_ratio is None and current_assets is None and current_liabilities is None and interest_expense is None:
            missing_notes.append("leverage data is incomplete")
            return None

        sub_scores: List[Optional[float]] = []
        if debt is not None and equity is not None and equity > 0:
            debt_equity = debt / equity
            sub_scores.append(min(1.0, debt_equity / 2.0))
        if current_ratio is not None:
            if current_ratio <= 1.0:
                sub_scores.append(1.0)
            elif current_ratio >= 2.0:
                sub_scores.append(0.0)
            else:
                sub_scores.append(1.0 - (current_ratio - 1.0))
        elif current_assets is not None and current_liabilities is not None and current_liabilities > 0:
            ratio = current_assets / current_liabilities
            if ratio <= 1.0:
                sub_scores.append(1.0)
            elif ratio >= 2.0:
                sub_scores.append(0.0)
            else:
                sub_scores.append(1.0 - (ratio - 1.0))
        if interest_expense is not None and ebit is not None and ebit > 0:
            coverage = ebit / interest_expense
            if coverage <= 1.0:
                sub_scores.append(1.0)
            elif coverage >= 10.0:
                sub_scores.append(0.0)
            else:
                sub_scores.append(1.0 - ((coverage - 1.0) / 9.0))
        if debt is not None and ebit is not None and ebit > 0:
            burden = debt / ebit
            sub_scores.append(min(1.0, burden / 4.0))

        return self._safely_average(sub_scores) or 0.0

    def _cash_flow_quality_risk(self, cash_flow: Mapping[str, Any], income: Mapping[str, Any], missing_notes: List[str]) -> Optional[float]:
        ocf = self._get_value(cash_flow, ["operating_cash_flow", "cash_flow_from_operations"])
        fcf = self._get_value(cash_flow, ["free_cash_flow"])
        revenue = self._get_value(income, ["revenue"])
        ocf_history = cash_flow.get("cash_flow_from_ops_history")
        fcf_history = cash_flow.get("free_cash_flow_history")

        if ocf is None and fcf is None and revenue is None:
            missing_notes.append("cash flow data is incomplete")
            return None

        sub_scores: List[Optional[float]] = []
        if ocf is not None and revenue is not None and revenue > 0:
            ocf_margin = ocf / revenue
            sub_scores.append(1.0 - self._normalize_ratio(ocf_margin, 0.0, 0.2))
        elif ocf is not None and revenue is None:
            sub_scores.append(0.5)
        if fcf is not None and revenue is not None and revenue > 0:
            fcf_margin = fcf / revenue
            sub_scores.append(1.0 - self._normalize_ratio(fcf_margin, 0.0, 0.15))
        elif fcf is not None and revenue is None:
            sub_scores.append(0.5)
        if isinstance(ocf_history, list) and ocf_history:
            negative_years = sum(1 for x in ocf_history if x is not None and x < 0)
            sub_scores.append(min(1.0, negative_years / len(ocf_history)))
        if isinstance(fcf_history, list) and fcf_history:
            negative_years = sum(1 for x in fcf_history if x is not None and x < 0)
            sub_scores.append(min(1.0, negative_years / len(fcf_history)))

        return self._safely_average(sub_scores) or 0.0

    def _valuation_stress_risk(self, market: Mapping[str, Any], missing_notes: List[str]) -> Optional[float]:
        pe = self._get_value(market, ["pe_ratio"])
        ps = self._get_value(market, ["ps_ratio"])
        peg = self._get_value(market, ["peg_ratio"])

        if pe is None and ps is None and peg is None:
            missing_notes.append("valuation data is incomplete")
            return None

        sub_scores: List[Optional[float]] = []
        if pe is not None:
            sub_scores.append(min(1.0, max(0.0, (pe - 10.0) / 15.0)))
        if ps is not None:
            sub_scores.append(min(1.0, max(0.0, (ps - 2.0) / 3.0)))
        if peg is not None:
            sub_scores.append(min(1.0, max(0.0, (peg - 1.0) / 2.0)))

        return self._safely_average(sub_scores) or 0.0

    def _growth_deceleration_risk(self, income: Mapping[str, Any], missing_notes: List[str]) -> Optional[float]:
        revenue_history = income.get("revenue_history")
        eps_history = income.get("eps_history")

        if not isinstance(revenue_history, list) and not isinstance(eps_history, list):
            missing_notes.append("growth data is incomplete")
            return None

        sub_scores: List[Optional[float]] = []
        if isinstance(revenue_history, list) and len(revenue_history) >= 2:
            latest = revenue_history[-1]
            prior = revenue_history[-2]
            if latest is not None and prior is not None and prior != 0:
                growth = (latest - prior) / abs(prior)
                previous = ((prior - revenue_history[-3]) / abs(revenue_history[-3])) if len(revenue_history) >= 3 and revenue_history[-3] not in (0, None) else growth
                if previous is not None:
                    slowdown = max(0.0, previous - growth)
                    sub_scores.append(min(1.0, slowdown / (abs(previous) + 1e-9)))
        if isinstance(eps_history, list) and len(eps_history) >= 2:
            latest = eps_history[-1]
            prior = eps_history[-2]
            if latest is not None and prior is not None and prior != 0:
                growth = (latest - prior) / abs(prior)
                previous = ((prior - eps_history[-3]) / abs(eps_history[-3])) if len(eps_history) >= 3 and eps_history[-3] not in (0, None) else growth
                if previous is not None:
                    slowdown = max(0.0, previous - growth)
                    sub_scores.append(min(1.0, slowdown / (abs(previous) + 1e-9)))

        return self._safely_average(sub_scores) or 0.0

    def _size_concentration_risk(self, market: Mapping[str, Any], profile: Mapping[str, Any], missing_notes: List[str]) -> Optional[float]:
        market_cap = self._get_value(market, ["market_cap"])
        customer_conc = self._get_value(profile, ["customer_concentration"])
        product_conc = self._get_value(profile, ["product_concentration"])

        if market_cap is None and customer_conc is None and product_conc is None:
            missing_notes.append("size and concentration data is incomplete")
            return None

        sub_scores: List[Optional[float]] = []
        if market_cap is not None:
            if market_cap < 1_000_000_000:
                sub_scores.append(1.0)
            elif market_cap < 10_000_000_000:
                sub_scores.append(0.5)
            else:
                sub_scores.append(0.0)
        if customer_conc is not None:
            sub_scores.append(min(1.0, max(0.0, customer_conc)))
        if product_conc is not None:
            sub_scores.append(min(1.0, max(0.0, product_conc)))

        return self._safely_average(sub_scores) or 0.0

    def _sector_policy_risk(self, profile: Mapping[str, Any], industry_metadata: Optional[Mapping[str, Any]], missing_notes: List[str]) -> Optional[float]:
        sector = profile.get("sector")
        if sector is None:
            missing_notes.append("sector metadata is missing")
            return None

        multiplier = 1.0
        if isinstance(industry_metadata, Mapping):
            sector_map = industry_metadata.get("sector_risk_multiplier")
            if isinstance(sector_map, Mapping) and sector in sector_map:
                try:
                    multiplier = float(sector_map[sector])
                except (TypeError, ValueError):
                    multiplier = 1.0

        base_risk = 0.5 if sector else 0.5
        risk = min(1.0, max(0.0, base_risk * multiplier))
        return risk

    def _altman_z_score_risk(self, financial_data: Mapping[str, Any], missing_notes: List[str]) -> Optional[float]:
        altman = self._get_value(financial_data, ["altman_z_score"])
        if altman is None:
            return None

        if altman >= 3.0:
            return 0.0
        if altman <= 1.8:
            return 1.0
        return min(1.0, max(0.0, (1.8 - altman) / (1.8 - 3.0)))

    @staticmethod
    def _normalize_value(raw: Optional[float], config: FactorConfig) -> float:
        if raw is None:
            return 0.0
        min_val = config.min_value
        max_val = config.max_value
        if max_val == min_val:
            return 0.0
        ratio = (raw - min_val) / (max_val - min_val)
        if config.direction == "lower_worse":
            ratio = 1.0 - ratio
        score = max(0.0, min(1.0, ratio)) * 100.0
        return round(score, 2)

    def _explanation(self, factors: Iterable[FactorResult]) -> str:
        active_count = sum(1 for f in factors if f.weight > 0 and f.raw_value is not None)
        return (
            f"The StructuralRiskModel aggregates {active_count} active risk components into a final score. "
            "Placeholder factors AAA, BBB, and CCC are present for future use and currently do not alter the score."
        )
