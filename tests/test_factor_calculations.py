import unittest

from models import StatisticalRiskModel, StructuralRiskModel


class TestStatisticalFactorCalculations(unittest.TestCase):
    def test_statistical_raw_factor_calculation_contains_expected_keys(self):
        model = StatisticalRiskModel()
        asset_series = [
            {"adj_close": 100.0, "volume": 1_000_000},
            {"adj_close": 102.0, "volume": 1_020_000},
            {"adj_close": 101.0, "volume": 990_000},
            {"adj_close": 103.0, "volume": 1_010_000},
        ]
        benchmark_series = [
            {"adj_close": 200.0, "volume": 3_000_000},
            {"adj_close": 201.0, "volume": 3_010_000},
            {"adj_close": 199.0, "volume": 2_990_000},
            {"adj_close": 202.0, "volume": 3_020_000},
        ]

        raw = model._calculate_raw_metrics(asset_series, benchmark_series)
        self.assertIn("annualized_volatility", raw)
        self.assertIn("beta", raw)
        self.assertIn("maximum_drawdown", raw)
        self.assertIn("AAA", raw)
        self.assertIsNone(raw["AAA"])


class TestStructuralFactorCalculations(unittest.TestCase):
    def test_structural_raw_factor_calculation_contains_expected_keys(self):
        model = StructuralRiskModel()
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

        raw = model._calculate_raw_metrics(financial_data, company_profile, {"sector_risk_multiplier": {"Technology": 1.0}}, [])
        self.assertIn("profitability_risk", raw)
        self.assertIn("leverage_risk", raw)
        self.assertIn("sector_policy_risk", raw)
        self.assertIn("BBB", raw)
        self.assertIsNone(raw["BBB"])


if __name__ == "__main__":
    unittest.main()
