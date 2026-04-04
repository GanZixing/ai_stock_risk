import unittest

from llm import LLMSummaryOutput
from models import StatisticalRiskModel, StructuralRiskModel
from schemas import RESERVED_FACTORS
from schemas.output_schema import build_agent_output


class TestOutputSchema(unittest.TestCase):
    def setUp(self):
        self.asset_history = [
            {"adj_close": price, "volume": 1000000 + idx * 5000}
            for idx, price in enumerate([150.0, 151.5, 149.0, 152.0, 153.0, 151.0])
        ]
        self.benchmark_history = [
            {"adj_close": price, "volume": 3000000 + idx * 10000}
            for idx, price in enumerate([420.0, 421.0, 419.0, 423.0, 425.0, 426.0])
        ]

        self.financial_data = {
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
        self.company_profile = {
            "sector": "Technology",
            "industry": "Software",
            "customer_concentration": 0.15,
            "product_concentration": 0.2,
        }
        self.industry_metadata = {"sector_risk_multiplier": {"Technology": 1.0}}

    def test_build_agent_output_schema(self):
        stat_result = StatisticalRiskModel().evaluate(
            ticker="AAPL",
            benchmark_ticker="SPY",
            lookback_window=6,
            price_history=self.asset_history,
            benchmark_history=self.benchmark_history,
        )
        struct_result = StructuralRiskModel().evaluate(
            ticker="AAPL",
            financial_data=self.financial_data,
            company_profile=self.company_profile,
            industry_metadata=self.industry_metadata,
        )
        summary = LLMSummaryOutput(
            short_summary="short",
            long_summary="long",
            key_risk_drivers=["Annualized Volatility"],
            model_disagreement_note="none",
            reserved_factor_note="AAA/BBB/CCC inactive",
            final_combined_interpretation="combined",
        )

        output = build_agent_output(
            ticker="AAPL",
            benchmark="SPY",
            statistical_result=stat_result,
            structural_result=struct_result,
            llm_summary=summary,
            timestamp="2026-04-04T00:00:00+00:00",
        )

        self.assertEqual(output["ticker"], "AAPL")
        self.assertEqual(output["benchmark"], "SPY")
        self.assertEqual(output["timestamp"], "2026-04-04T00:00:00+00:00")
        self.assertIn("statistical_risk", output)
        self.assertIn("structural_risk", output)
        self.assertIn("llm_summary", output)
        self.assertIn("reserved_factors_status", output)

        for model_key in ("statistical_risk", "structural_risk"):
            model_output = output[model_key]
            self.assertIn("model_name", model_output)
            self.assertIn("score", model_output)
            self.assertIn("level", model_output)
            self.assertIn("factor_scores", model_output)
            self.assertIn("raw_metrics", model_output)
            self.assertIn("weights", model_output)
            self.assertIn("notes", model_output)

        for factor in RESERVED_FACTORS:
            reserved = output["reserved_factors_status"][factor]
            self.assertEqual(reserved["enabled"], False)
            self.assertEqual(reserved["weight"], 0.0)
            self.assertEqual(reserved["normalized_score"], 0.0)
            self.assertEqual(reserved["affects_final_score"], False)


if __name__ == "__main__":
    unittest.main()
