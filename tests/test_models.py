import unittest

from models import StatisticalRiskModel, StructuralRiskModel
from schemas import RESERVED_FACTORS


class TestStatisticalRiskModel(unittest.TestCase):
    def setUp(self):
        self.model = StatisticalRiskModel()
        self.asset_history = [
            {"adj_close": price, "volume": 1000000 + idx * 10000}
            for idx, price in enumerate([100, 101, 102, 100, 98, 99, 100, 102, 101, 103])
        ]
        self.benchmark_history = [
            {"adj_close": price, "volume": 3000000 + idx * 20000}
            for idx, price in enumerate([200, 202, 204, 203, 205, 207, 206, 208, 210, 212])
        ]

    def test_evaluate_returns_expected_structure(self):
        result = self.model.evaluate(
            ticker="TEST",
            benchmark_ticker="SPY",
            lookback_window=10,
            price_history=self.asset_history,
            benchmark_history=self.benchmark_history,
        )

        self.assertEqual(result.model_name, "StatisticalRiskModel")
        self.assertIn(result.risk_level, {"Low", "Moderate", "High", "Extreme"})
        self.assertGreaterEqual(result.overall_score, 0)
        self.assertLessEqual(result.overall_score, 100)
        self.assertIsInstance(result.factor_breakdown, list)
        self.assertEqual(len(result.factor_breakdown), len(result.weights_used))
        self.assertIn("historical_cvar_95", result.raw_metrics)
        self.assertIn("liquidity_risk_proxy", result.raw_metrics)

    def test_placeholders_do_not_affect_score(self):
        result = self.model.evaluate(
            ticker="TEST",
            benchmark_ticker="SPY",
            lookback_window=10,
            price_history=self.asset_history,
            benchmark_history=self.benchmark_history,
        )

        placeholder_scores = [f.normalized_score for f in result.factor_breakdown if f.factor in {"AAA", "BBB", "CCC"}]
        placeholder_weights = [f.weight for f in result.factor_breakdown if f.factor in {"AAA", "BBB", "CCC"}]
        self.assertEqual(placeholder_scores, [0.0, 0.0, 0.0])
        self.assertEqual(placeholder_weights, [0.0, 0.0, 0.0])
        for factor in RESERVED_FACTORS:
            self.assertIn(factor, result.reserved_factors)
            self.assertEqual(result.reserved_factors[factor]["weight"], 0.0)
            self.assertEqual(result.reserved_factors[factor]["normalized_score"], 0.0)
            self.assertFalse(result.reserved_factors[factor]["active"])


class TestStructuralRiskModel(unittest.TestCase):
    def setUp(self):
        self.model = StructuralRiskModel()
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

    def test_evaluate_structure_returns_expected_result(self):
        result = self.model.evaluate(
            ticker="TEST",
            financial_data=self.financial_data,
            company_profile=self.company_profile,
            industry_metadata=self.industry_metadata,
        )

        self.assertEqual(result.model_name, "StructuralRiskModel")
        self.assertIn(result.risk_level, {"Low", "Moderate", "High", "Extreme"})
        self.assertGreaterEqual(result.overall_score, 0)
        self.assertLessEqual(result.overall_score, 100)
        self.assertIsInstance(result.factor_breakdown, list)
        self.assertEqual(len(result.factor_breakdown), len(result.weights_used))
        self.assertIn("profitability_risk", result.raw_metrics)
        self.assertIn("AAA", result.raw_metrics)

    def test_placeholders_do_not_affect_score(self):
        result = self.model.evaluate(
            ticker="TEST",
            financial_data=self.financial_data,
            company_profile=self.company_profile,
            industry_metadata=self.industry_metadata,
        )

        placeholders = [f for f in result.factor_breakdown if f.factor in {"AAA", "BBB", "CCC"}]
        self.assertEqual([f.normalized_score for f in placeholders], [0.0, 0.0, 0.0])
        self.assertEqual([f.weight for f in placeholders], [0.0, 0.0, 0.0])
        for factor in RESERVED_FACTORS:
            self.assertIn(factor, result.reserved_factors)
            self.assertEqual(result.reserved_factors[factor]["weight"], 0.0)
            self.assertEqual(result.reserved_factors[factor]["normalized_score"], 0.0)
            self.assertFalse(result.reserved_factors[factor]["active"])

    def test_missing_data_is_handled_gracefully(self):
        partial_data = {
            "income_statement": {
                "net_income": 50.0,
                "revenue": 600.0,
            }
        }

        result = self.model.evaluate(
            ticker="TEST",
            financial_data=partial_data,
            company_profile={"sector": "Technology"},
        )

        self.assertGreaterEqual(result.overall_score, 0)
        self.assertLessEqual(result.overall_score, 100)


if __name__ == "__main__":
    unittest.main()
