from llm_summary_layer import LLMSummaryLayer
from statistical_risk_model import StatisticalRiskModel
from structural_risk_model import StructuralRiskModel


def main() -> None:
    # Sample data for demonstration
    asset_history = [
        {"adj_close": price, "volume": 1000000 + idx * 5000}
        for idx, price in enumerate([150.0, 151.5, 149.0, 152.0, 153.0, 151.0])
    ]
    benchmark_history = [
        {"adj_close": price, "volume": 3000000 + idx * 10000}
        for idx, price in enumerate([420.0, 421.0, 419.0, 423.0, 425.0, 426.0])
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
        price_history=asset_history,
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
        statistical_result=stat_result.to_dict(),
        structural_result=struct_result.to_dict(),
        reserved_factors_status={"AAA": "placeholder", "BBB": "placeholder", "CCC": "placeholder"},
    )

    print("Short Summary:", summary.short_summary)
    print("Long Summary:", summary.long_summary)
    print("Key Risk Drivers:", summary.key_risk_drivers)
    print("Model Disagreement Note:", summary.model_disagreement_note)
    print("Reserved Factor Note:", summary.reserved_factor_note)
    print("Final Combined Interpretation:", summary.final_combined_interpretation)


if __name__ == "__main__":
    main()
