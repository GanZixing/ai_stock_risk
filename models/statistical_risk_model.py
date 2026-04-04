from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from config import ConfigLoader
from data import PriceDataAdapter
from normalization import normalize_value
from schemas import (
    BaseModel,
    FactorConfig,
    FactorResult,
    ModelConfig,
    ModelResult,
    apply_reserved_factor_values,
    get_reserved_factor_status,
)
from scoring import aggregate_score, risk_level_label


class StatisticalRiskModel(BaseModel):
    model_name = "StatisticalRiskModel"

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ConfigLoader.load_default_statistical_config()

    def evaluate(
        self,
        ticker: str,
        benchmark_ticker: str = "SPY",
        lookback_window: int = 252,
        price_history: Optional[Sequence[Mapping[str, Any]]] = None,
        benchmark_history: Optional[Sequence[Mapping[str, Any]]] = None,
        data_adapter: Optional[PriceDataAdapter] = None,
        reserved_factor_overrides: Optional[Mapping[str, Optional[float]]] = None,
    ) -> ModelResult:
        if data_adapter is not None:
            data = data_adapter.get_price_history(ticker, lookback_window, benchmark_ticker)
            price_history = price_history or data.get("asset")
            benchmark_history = benchmark_history or data.get("benchmark")

        if price_history is None or benchmark_history is None:
            raise ValueError("Both asset and benchmark price history must be provided.")

        asset_series = self._extract_series(price_history)
        benchmark_series = self._extract_series(benchmark_history)

        raw_metrics = self._calculate_raw_metrics(
            asset_series,
            benchmark_series,
            reserved_factor_overrides=reserved_factor_overrides,
        )
        factor_breakdown = self._normalize_factors(raw_metrics)
        overall_score = aggregate_score(factor_breakdown)
        risk_level = risk_level_label(overall_score)

        return ModelResult(
            model_name=self.model_name,
            overall_score=round(overall_score, 2),
            risk_level=risk_level,
            factor_breakdown=factor_breakdown,
            raw_metrics=raw_metrics,
            weights_used={factor: cfg.weight for factor, cfg in self.config.factor_thresholds.items()},
            reserved_factors=get_reserved_factor_status(raw_metrics, self.config.factor_thresholds),
        )

    def _calculate_raw_metrics(
        self,
        asset_series: Sequence[Mapping[str, Any]],
        benchmark_series: Sequence[Mapping[str, Any]],
        reserved_factor_overrides: Optional[Mapping[str, Optional[float]]] = None,
    ) -> Dict[str, Optional[float]]:
        asset_prices = self._get_price_values(asset_series)
        benchmark_prices = self._get_price_values(benchmark_series)

        asset_returns = self._daily_returns(asset_prices)
        benchmark_returns = self._daily_returns(benchmark_prices)

        raw_metrics = {
            "annualized_volatility": self._annualized_volatility(asset_returns),
            "downside_volatility": self._downside_volatility(asset_returns),
            "beta": self._beta(asset_returns, benchmark_returns),
            "maximum_drawdown": self._maximum_drawdown(asset_prices),
            "historical_var_95": self._historical_var(asset_returns, 0.95),
            "historical_cvar_95": self._historical_cvar(asset_returns, 0.95),
            "liquidity_risk_proxy": self._liquidity_risk_proxy(asset_series),
        }
        # Reserved placeholders are always present, with optional override hook for future activation.
        return apply_reserved_factor_values(raw_metrics, reserved_factor_overrides)

    def _normalize_factors(self, raw_metrics: Dict[str, Optional[float]]) -> List[FactorResult]:
        results: List[FactorResult] = []
        for factor, cfg in self.config.factor_thresholds.items():
            raw_value = raw_metrics.get(factor)
            normalized_score = normalize_value(raw_value, cfg)
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
    def _extract_series(
        data: Sequence[Mapping[str, Any]]
    ) -> List[Mapping[str, Any]]:
        if not isinstance(data, Sequence):
            raise ValueError("Price history must be a sequence of data rows.")
        cleaned = [row for row in data if row is not None]
        if not cleaned:
            raise ValueError("Price history cannot be empty.")
        return cleaned
