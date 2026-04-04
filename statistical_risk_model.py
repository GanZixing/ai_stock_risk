from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


class RiskDataAdapter(ABC):
    """Adapter interface for a historical price data provider."""

    @abstractmethod
    def get_price_history(
        self,
        ticker: str,
        lookback_window: int,
        benchmark_ticker: Optional[str] = None,
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Return a mapping with at least 'asset' and 'benchmark' time series."""
        raise NotImplementedError


@dataclass
class FactorConfig:
    weight: float
    min_value: float
    max_value: float
    direction: str = "higher_worse"


@dataclass
class StatisticalRiskModelConfig:
    factor_thresholds: Dict[str, FactorConfig] = field(default_factory=dict)

    @staticmethod
    def default() -> "StatisticalRiskModelConfig":
        return StatisticalRiskModelConfig(
            factor_thresholds={
                "annualized_volatility": FactorConfig(weight=15, min_value=0.10, max_value=0.50, direction="higher_worse"),
                "downside_volatility": FactorConfig(weight=15, min_value=0.05, max_value=0.35, direction="higher_worse"),
                "beta": FactorConfig(weight=10, min_value=0.20, max_value=2.00, direction="higher_worse"),
                "maximum_drawdown": FactorConfig(weight=15, min_value=0.05, max_value=0.60, direction="higher_worse"),
                "historical_var_95": FactorConfig(weight=10, min_value=0.01, max_value=0.10, direction="higher_worse"),
                "historical_cvar_95": FactorConfig(weight=10, min_value=0.015, max_value=0.12, direction="higher_worse"),
                "liquidity_risk_proxy": FactorConfig(weight=10, min_value=0.0, max_value=2.5e-06, direction="higher_worse"),
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
class StatisticalRiskModelResult:
    model_name: str
    overall_score: float
    risk_level: str
    factor_breakdown: List[FactorResult]
    raw_metrics: Dict[str, Optional[float]]
    weights_used: Dict[str, float]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["overall_score"] = float(self.overall_score)
        return result


class StatisticalRiskModel:
    model_name = "StatisticalRiskModel"

    def __init__(self, config: Optional[StatisticalRiskModelConfig] = None):
        self.config = config or StatisticalRiskModelConfig.default()

    def evaluate(
        self,
        ticker: str,
        benchmark_ticker: str = "SPY",
        lookback_window: int = 252,
        price_history: Optional[Sequence[Mapping[str, Any]]] = None,
        benchmark_history: Optional[Sequence[Mapping[str, Any]]] = None,
        data_adapter: Optional[RiskDataAdapter] = None,
    ) -> StatisticalRiskModelResult:
        if data_adapter is not None:
            data = data_adapter.get_price_history(ticker, lookback_window, benchmark_ticker)
            price_history = price_history or data.get("asset")
            benchmark_history = benchmark_history or data.get("benchmark")

        if price_history is None or benchmark_history is None:
            raise ValueError("Both asset and benchmark price history must be provided.")

        asset_series = self._extract_series(price_history)
        benchmark_series = self._extract_series(benchmark_history)

        raw_metrics = self._calculate_raw_metrics(asset_series, benchmark_series)
        factor_breakdown = self._normalize_factors(raw_metrics)
        overall_score = self._aggregate_score(factor_breakdown)
        risk_level = self._risk_level_label(overall_score)

        return StatisticalRiskModelResult(
            model_name=self.model_name,
            overall_score=round(overall_score, 2),
            risk_level=risk_level,
            factor_breakdown=factor_breakdown,
            raw_metrics=raw_metrics,
            weights_used={factor: cfg.weight for factor, cfg in self.config.factor_thresholds.items()},
            explanation=self._build_explanation(factor_breakdown),
        )

    def _calculate_raw_metrics(
        self,
        asset_series: Sequence[Mapping[str, Any]],
        benchmark_series: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Optional[float]]:
        asset_prices = self._get_price_values(asset_series)
        benchmark_prices = self._get_price_values(benchmark_series)

        asset_returns = self._daily_returns(asset_prices)
        benchmark_returns = self._daily_returns(benchmark_prices)

        return {
            "annualized_volatility": self._annualized_volatility(asset_returns),
            "downside_volatility": self._downside_volatility(asset_returns),
            "beta": self._beta(asset_returns, benchmark_returns),
            "maximum_drawdown": self._maximum_drawdown(asset_prices),
            "historical_var_95": self._historical_var(asset_returns, 0.95),
            "historical_cvar_95": self._historical_cvar(asset_returns, 0.95),
            "liquidity_risk_proxy": self._liquidity_risk_proxy(asset_series),
            "AAA": None,
            "BBB": None,
            "CCC": None,
        }

    def _normalize_factors(self, raw_metrics: Dict[str, Optional[float]]) -> List[FactorResult]:
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
    def _build_explanation(factors: Iterable[FactorResult]) -> str:
        active_factors = [f for f in factors if f.weight > 0]
        return (
            "The overall risk score is a weighted average of normalized risk factors. "
            f"{len(active_factors)} active factors contributed to the score. "
            "AAA, BBB, and CCC are configured as placeholders and currently do not affect the final score."
        )

    @staticmethod
    def _get_price_values(series: Sequence[Mapping[str, Any]]) -> List[float]:
        prices: List[float] = []
        for row in series:
            if row is None:
                continue
            price = None
            for key in ("adj_close", "adjclose", "close", "price"):
                if key in row and row[key] is not None:
                    price = row[key]
                    break
            if price is None:
                raise ValueError("Price series must include 'adj_close', 'close', or 'price'.")
            prices.append(float(price))
        if len(prices) < 2:
            raise ValueError("Price series must contain at least two observations.")
        return prices

    @staticmethod
    def _daily_returns(prices: Sequence[float]) -> List[float]:
        return [
            (prices[i] - prices[i - 1]) / prices[i - 1]
            for i in range(1, len(prices))
            if prices[i - 1] != 0
        ]

    @staticmethod
    def _annualized_volatility(returns: Sequence[float]) -> Optional[float]:
        if len(returns) < 2:
            return None
        return stdev(returns) * math.sqrt(252)

    @staticmethod
    def _downside_volatility(returns: Sequence[float]) -> Optional[float]:
        downside = [r for r in returns if r < 0]
        if len(downside) < 2:
            return 0.0
        return stdev(downside) * math.sqrt(252)

    @staticmethod
    def _beta(asset_returns: Sequence[float], benchmark_returns: Sequence[float]) -> Optional[float]:
        if len(asset_returns) < 2 or len(benchmark_returns) < 2:
            return None
        paired = list(zip(asset_returns[-len(benchmark_returns) :], benchmark_returns))
        if len(paired) < 2:
            return None
        asset_vals, benchmark_vals = zip(*paired)
        benchmark_mean = mean(benchmark_vals)
        asset_mean = mean(asset_vals)
        covariance = sum((a - asset_mean) * (b - benchmark_mean) for a, b in zip(asset_vals, benchmark_vals)) / (len(paired) - 1)
        variance = sum((b - benchmark_mean) ** 2 for b in benchmark_vals) / (len(paired) - 1)
        if variance == 0:
            return None
        return covariance / variance

    @staticmethod
    def _maximum_drawdown(prices: Sequence[float]) -> Optional[float]:
        peak = prices[0]
        max_drawdown = 0.0
        for price in prices:
            peak = max(peak, price)
            if peak > 0:
                drawdown = (peak - price) / peak
                max_drawdown = max(max_drawdown, drawdown)
        return max_drawdown

    @staticmethod
    def _historical_var(returns: Sequence[float], confidence_level: float) -> Optional[float]:
        if not returns:
            return None
        sorted_returns = sorted(returns)
        index = max(0, min(len(sorted_returns) - 1, int((1 - confidence_level) * len(sorted_returns))))
        var = sorted_returns[index]
        return abs(var)

    @staticmethod
    def _historical_cvar(returns: Sequence[float], confidence_level: float) -> Optional[float]:
        if not returns:
            return None
        sorted_returns = sorted(returns)
        threshold_index = max(0, int((1 - confidence_level) * len(sorted_returns)))
        tail = sorted_returns[: threshold_index + 1]
        if not tail:
            return 0.0
        return abs(mean(tail))

    @staticmethod
    def _liquidity_risk_proxy(series: Sequence[Mapping[str, Any]]) -> Optional[float]:
        volume_values: List[float] = []
        dollar_volumes: List[float] = []
        for row in series:
            price = None
            volume = None
            for key in ("adj_close", "adjclose", "close", "price"):
                if key in row and row[key] is not None:
                    price = float(row[key])
                    break
            for key in ("volume", "vol"):
                if key in row and row[key] is not None:
                    volume = float(row[key])
                    break
            if price is None or volume is None:
                continue
            if volume > 0:
                dollar_volumes.append(price * volume)
                volume_values.append(volume)
        if not dollar_volumes:
            return 0.0
        average_dollar_volume = mean(dollar_volumes)
        if average_dollar_volume <= 0:
            return 0.0
        return 1.0 / average_dollar_volume

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

    @staticmethod
    def _extract_series(
        data: Sequence[Mapping[str, Any]]
    ) -> List[Mapping[str, Any]]:
        if not isinstance(data, Sequence):
            raise ValueError("Price history must be a sequence of data rows.")
        cleaned = [row for row in data if row is not None]
        if not cleaned:
            raise ValueError("Price history cannot be empty.")
        return cleaned
