#!/usr/bin/env python3

"""
Sample run command for the US Stock Risk Agent.

This script demonstrates running both models and the LLM summary layer.
"""

import json
from models import StatisticalRiskModel, StructuralRiskModel
from llm import LLMSummaryLayer
from schemas.output_schema import build_agent_output

def main():
    # Sample data
    price_history = [
        {"adj_close": 150.0, "volume": 1000000},
        {"adj_close": 151.5, "volume": 1010000},
        {"adj_close": 149.0, "volume": 980000},
        {"adj_close": 152.0, "volume": 1020000},
        {"adj_close": 153.0, "volume": 1030000},
        {"adj_close": 151.0, "volume": 990000},
    ]
    benchmark_history = [
        {"adj_close": 420.0, "volume": 3000000},
        {"adj_close": 421.0, "volume": 3050000},
        {"adj_close": 419.0, "volume": 2990000},
        {"adj_close": 423.0, "volume": 3060000},
        {"adj_close": 425.0, "volume": 3080000},
        {"adj_close": 426.0, "volume": 3070000},
    ]

    financial_data = {
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
    }
    company_profile = {
        "sector": "Technology",
        "industry": "Software",
        "customer_concentration": 0.15,
        "product_concentration": 0.2,
    }
    industry_metadata = {"sector_risk_multiplier": {"Technology": 1.0}}

    # Run models
    stat_model = StatisticalRiskModel()
    stat_result = stat_model.evaluate(
        ticker="AAPL",
        benchmark_ticker="SPY",
        lookback_window=6,
        price_history=price_history,
        benchmark_history=benchmark_history,
    )

    struct_model = StructuralRiskModel()
    struct_result = struct_model.evaluate(
        ticker="AAPL",
        financial_data=financial_data,
        company_profile=company_profile,
        industry_metadata=industry_metadata,
    )

    # Generate LLM summary
    summary_layer = LLMSummaryLayer()
    summary = summary_layer.generate_summary(
        ticker="AAPL",
        statistical_result=stat_result,
        structural_result=struct_result,
        reserved_factors_status={"AAA": "placeholder", "BBB": "placeholder", "CCC": "placeholder"},
    )

    # Output using standardized schema.
    output = build_agent_output(
        ticker="AAPL",
        benchmark="SPY",
        statistical_result=stat_result,
        structural_result=struct_result,
        llm_summary=summary,
    )

    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
