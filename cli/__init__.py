#!/usr/bin/env python3

import argparse
import json
from typing import Any, Dict

from config import ConfigLoader
from llm import LLMSummaryLayer
from models import StatisticalRiskModel, StructuralRiskModel
from schemas.output_schema import build_agent_output


def main():
    parser = argparse.ArgumentParser(description="US Stock Risk Agent")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark ticker (default: SPY)")
    parser.add_argument("--lookback", type=int, default=252, help="Lookback window for statistical model")
    parser.add_argument("--price-data", help="Path to JSON file with price history")
    parser.add_argument("--financial-data", help="Path to JSON file with financial data")
    parser.add_argument("--company-profile", help="Path to JSON file with company profile")
    parser.add_argument("--industry-metadata", help="Path to JSON file with industry metadata")
    parser.add_argument("--config-stat", help="Path to statistical model config JSON")
    parser.add_argument("--config-struct", help="Path to structural model config JSON")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--llm-summary", action="store_true", help="Generate LLM summary")

    args = parser.parse_args()

    # Load data
    price_history = None
    benchmark_history = None
    if args.price_data:
        with open(args.price_data, 'r') as f:
            data = json.load(f)
            price_history = data.get("asset")
            benchmark_history = data.get("benchmark")

    financial_data = None
    if args.financial_data:
        with open(args.financial_data, 'r') as f:
            financial_data = json.load(f)

    company_profile = None
    if args.company_profile:
        with open(args.company_profile, 'r') as f:
            company_profile = json.load(f)

    industry_metadata = None
    if args.industry_metadata:
        with open(args.industry_metadata, 'r') as f:
            industry_metadata = json.load(f)

    # Load configs
    stat_config = ConfigLoader.load_default_statistical_config()
    if args.config_stat:
        stat_config = ConfigLoader.load_from_file(args.config_stat)

    struct_config = ConfigLoader.load_default_structural_config()
    if args.config_struct:
        struct_config = ConfigLoader.load_from_file(args.config_struct)

    # Run models
    stat_model = StatisticalRiskModel(stat_config)
    stat_result = stat_model.evaluate(
        ticker=args.ticker,
        benchmark_ticker=args.benchmark,
        lookback_window=args.lookback,
        price_history=price_history,
        benchmark_history=benchmark_history,
    )

    struct_model = StructuralRiskModel(struct_config)
    struct_result = struct_model.evaluate(
        ticker=args.ticker,
        financial_data=financial_data,
        company_profile=company_profile,
        industry_metadata=industry_metadata,
    )

    summary = None
    if args.llm_summary:
        summary_layer = LLMSummaryLayer()
        summary = summary_layer.generate_summary(
            ticker=args.ticker,
            statistical_result=stat_result,
            structural_result=struct_result,
        )

    # Prepare top-level agent-friendly output schema.
    output: Dict[str, Any] = build_agent_output(
        ticker=args.ticker,
        benchmark=args.benchmark,
        statistical_result=stat_result,
        structural_result=struct_result,
        llm_summary=summary,
    )

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
