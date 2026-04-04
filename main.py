from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from config import ConfigLoader
from llm import LLMSummaryLayer
from models import StatisticalRiskModel, StructuralRiskModel
from schemas import FactorConfig, ModelConfig, RESERVED_FACTORS
from schemas.output_schema import build_agent_output


def _load_json_file(file_path: str) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _model_config_from_dict(config_data: Optional[Mapping[str, Any]]) -> Optional[ModelConfig]:
    if not config_data:
        return None

    threshold_map = config_data.get("factor_thresholds", config_data)
    parsed: Dict[str, FactorConfig] = {}
    for factor, cfg in threshold_map.items():
        if isinstance(cfg, FactorConfig):
            parsed[factor] = cfg
        else:
            parsed[factor] = FactorConfig(
                weight=float(cfg.get("weight", 0.0)),
                min_value=float(cfg.get("min_value", 0.0)),
                max_value=float(cfg.get("max_value", 1.0)),
                direction=cfg.get("direction", "higher_worse"),
            )

    # Ensure reserved placeholders are always present in runtime config.
    for reserved_factor in RESERVED_FACTORS:
        parsed.setdefault(
            reserved_factor,
            FactorConfig(weight=0.0, min_value=0.0, max_value=1.0, direction="higher_worse"),
        )

    return ModelConfig(factor_thresholds=parsed)


def _default_runtime_data() -> Dict[str, Any]:
    return {
        "price_history": [
            {"adj_close": 150.0, "volume": 1000000},
            {"adj_close": 151.5, "volume": 1010000},
            {"adj_close": 149.0, "volume": 980000},
            {"adj_close": 152.0, "volume": 1020000},
            {"adj_close": 153.0, "volume": 1030000},
            {"adj_close": 151.0, "volume": 990000},
        ],
        "benchmark_history": [
            {"adj_close": 420.0, "volume": 3000000},
            {"adj_close": 421.0, "volume": 3050000},
            {"adj_close": 419.0, "volume": 2990000},
            {"adj_close": 423.0, "volume": 3060000},
            {"adj_close": 425.0, "volume": 3080000},
            {"adj_close": 426.0, "volume": 3070000},
        ],
        "financial_data": {
            "income_statement": {
                "net_income": 120.0,
                "revenue": 1000.0,
                "gross_margin": 0.35,
                "operating_margin": 0.18,
                "roe": 0.15,
                "roa": 0.08,
                "revenue_history": [900.0, 950.0, 1000.0],
                "eps_history": [2.0, 2.3, 2.5],
                "net_income_history": [100.0, 110.0, 120.0],
            },
            "balance_sheet": {
                "total_debt": 300.0,
                "total_equity": 900.0,
                "current_assets": 500.0,
                "current_liabilities": 250.0,
                "interest_expense": 15.0,
                "ebit": 80.0,
            },
            "cash_flow": {
                "operating_cash_flow": 150.0,
                "free_cash_flow": 90.0,
                "cash_flow_from_ops_history": [130.0, 140.0, 150.0],
                "free_cash_flow_history": [80.0, 85.0, 90.0],
            },
            "market": {
                "market_cap": 12_000_000_000.0,
                "pe_ratio": 18.0,
                "ps_ratio": 3.0,
                "peg_ratio": 1.8,
            },
        },
        "company_profile": {
            "sector": "Technology",
            "industry": "Software",
            "customer_concentration": 0.15,
            "product_concentration": 0.2,
        },
        "industry_metadata": {"sector_risk_multiplier": {"Technology": 1.0}},
    }


def _load_runtime_payload(config_path: Optional[str]) -> Tuple[Dict[str, Any], Optional[ModelConfig], Optional[ModelConfig]]:
    if not config_path:
        return _default_runtime_data(), None, None

    payload = _load_json_file(config_path)
    data_section = payload.get("data", payload)
    statistical_cfg = _model_config_from_dict(payload.get("statistical_config"))
    structural_cfg = _model_config_from_dict(payload.get("structural_config"))
    return {
        "price_history": data_section.get("price_history"),
        "benchmark_history": data_section.get("benchmark_history"),
        "financial_data": data_section.get("financial_data"),
        "company_profile": data_section.get("company_profile"),
        "industry_metadata": data_section.get("industry_metadata"),
    }, statistical_cfg, structural_cfg


def run_stock_risk_agent(
    ticker: str,
    benchmark: str,
    lookback: int,
    disable_llm: bool,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    runtime_data, stat_cfg_override, struct_cfg_override = _load_runtime_payload(config_path)

    stat_config = stat_cfg_override or ConfigLoader.load_default_statistical_config()
    struct_config = struct_cfg_override or ConfigLoader.load_default_structural_config()

    stat_model = StatisticalRiskModel(config=stat_config)
    stat_result = stat_model.evaluate(
        ticker=ticker,
        benchmark_ticker=benchmark,
        lookback_window=lookback,
        price_history=runtime_data.get("price_history"),
        benchmark_history=runtime_data.get("benchmark_history"),
    )

    struct_model = StructuralRiskModel(config=struct_config)
    struct_result = struct_model.evaluate(
        ticker=ticker,
        financial_data=runtime_data.get("financial_data"),
        company_profile=runtime_data.get("company_profile"),
        industry_metadata=runtime_data.get("industry_metadata"),
    )

    llm_summary = None
    if not disable_llm:
        summary_layer = LLMSummaryLayer()
        llm_summary = summary_layer.generate_summary(
            ticker=ticker,
            statistical_result=stat_result,
            structural_result=struct_result,
            reserved_factors_status={factor: "inactive_placeholder" for factor in RESERVED_FACTORS},
        )

    return build_agent_output(
        ticker=ticker,
        benchmark=benchmark,
        statistical_result=stat_result,
        structural_result=struct_result,
        llm_summary=llm_summary,
    )


def _print_formatted_summary(output: Dict[str, Any], llm_enabled: bool) -> None:
    stat = output["statistical_risk"]
    struct = output["structural_risk"]

    print(f"Ticker: {output['ticker']} | Benchmark: {output['benchmark']}")
    print(f"Timestamp: {output['timestamp']}")
    print("-")
    print(f"Statistical Risk -> score: {stat['score']:.2f}, level: {stat['level']}")
    print(f"Structural Risk  -> score: {struct['score']:.2f}, level: {struct['level']}")

    if llm_enabled and output.get("llm_summary"):
        llm_summary = output["llm_summary"]
        print("-")
        print("LLM Summary:")
        print(f"  Short: {llm_summary['short_summary']}")
        print(f"  Disagreement: {llm_summary['model_disagreement_note']}")

    print("-")
    print("Reserved Factors:")
    for factor in RESERVED_FACTORS:
        item = output["reserved_factors_status"][factor]
        print(
            f"  {factor}: enabled={item['enabled']}, weight={item['weight']}, "
            f"normalized_score={item['normalized_score']}, affects_final_score={item['affects_final_score']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock Risk Agent CLI")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol (e.g., TSLA)")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark ticker (default: SPY)")
    parser.add_argument("--lookback", type=int, default=252, help="Lookback window for statistical model")
    parser.add_argument("--output-json", help="Optional output JSON file path")
    parser.add_argument("--disable-llm", action="store_true", help="Disable LLM summary generation")
    parser.add_argument(
        "--config",
        help="Optional config JSON path containing data and/or model config overrides",
    )

    args = parser.parse_args()

    output = run_stock_risk_agent(
        ticker=args.ticker,
        benchmark=args.benchmark,
        lookback=args.lookback,
        disable_llm=args.disable_llm,
        config_path=args.config,
    )

    _print_formatted_summary(output, llm_enabled=not args.disable_llm)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Exported JSON to {args.output_json}")


if __name__ == "__main__":
    main()
