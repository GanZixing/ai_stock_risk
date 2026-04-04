import unittest

from structural_risk_model import StructuralRiskModel


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
        self.assertEqual(result.missing_data_notes, [])

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

        self.assertIsInstance(result.missing_data_notes, list)
        self.assertTrue(len(result.missing_data_notes) >= 1)
        self.assertGreaterEqual(result.overall_score, 0)
        self.assertLessEqual(result.overall_score, 100)


if __name__ == "__main__":
    unittest.main()
