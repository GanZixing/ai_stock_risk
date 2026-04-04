from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from schemas import FactorConfig, ModelConfig, get_reserved_factor_config


class ConfigLoader:
    @staticmethod
    def load_from_file(file_path: str) -> ModelConfig:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        with open(path, 'r') as f:
            data = json.load(f)

        thresholds = {}
        for factor, cfg in data.get("factor_thresholds", {}).items():
            if isinstance(cfg, FactorConfig):
                thresholds[factor] = cfg
            else:
                thresholds[factor] = FactorConfig(
                    weight=float(cfg.get("weight", 0.0)),
                    min_value=float(cfg.get("min_value", 0.0)),
                    max_value=float(cfg.get("max_value", 1.0)),
                    direction=cfg.get("direction", "higher_worse"),
                )

        # Reserved factors must always exist, even if missing from file config.
        for factor, default_cfg in get_reserved_factor_config().items():
            thresholds.setdefault(factor, default_cfg)

        return ModelConfig(factor_thresholds=thresholds)

    @staticmethod
    def load_default_statistical_config() -> ModelConfig:
        thresholds = {
            "annualized_volatility": FactorConfig(weight=15, min_value=0.10, max_value=0.50, direction="higher_worse"),
            "downside_volatility": FactorConfig(weight=15, min_value=0.05, max_value=0.35, direction="higher_worse"),
            "beta": FactorConfig(weight=10, min_value=0.20, max_value=2.00, direction="higher_worse"),
            "maximum_drawdown": FactorConfig(weight=15, min_value=0.05, max_value=0.60, direction="higher_worse"),
            "historical_var_95": FactorConfig(weight=10, min_value=0.01, max_value=0.10, direction="higher_worse"),
            "historical_cvar_95": FactorConfig(weight=10, min_value=0.015, max_value=0.12, direction="higher_worse"),
            "liquidity_risk_proxy": FactorConfig(weight=10, min_value=0.0, max_value=2.5e-06, direction="higher_worse"),
        }
        thresholds.update(get_reserved_factor_config())
        return ModelConfig(factor_thresholds=thresholds)

    @staticmethod
    def load_default_structural_config() -> ModelConfig:
        thresholds = {
            "profitability_risk": FactorConfig(weight=15, min_value=0.0, max_value=1.0, direction="higher_worse"),
            "leverage_risk": FactorConfig(weight=15, min_value=0.0, max_value=1.0, direction="higher_worse"),
            "cash_flow_quality_risk": FactorConfig(weight=15, min_value=0.0, max_value=1.0, direction="higher_worse"),
            "valuation_stress_risk": FactorConfig(weight=10, min_value=0.0, max_value=1.0, direction="higher_worse"),
            "growth_deceleration_risk": FactorConfig(weight=10, min_value=0.0, max_value=1.0, direction="higher_worse"),
            "size_concentration_risk": FactorConfig(weight=10, min_value=0.0, max_value=1.0, direction="higher_worse"),
            "sector_policy_risk": FactorConfig(weight=10, min_value=0.0, max_value=1.0, direction="higher_worse"),
            "altman_z_score": FactorConfig(weight=5, min_value=0.0, max_value=1.0, direction="higher_worse"),
        }
        thresholds.update(get_reserved_factor_config())
        return ModelConfig(factor_thresholds=thresholds)
