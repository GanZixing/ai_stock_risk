import unittest

from llm import LLMSummaryLayer, LLMSummaryOutput


class TestLLMSummaryLayer(unittest.TestCase):
    def setUp(self):
        self.layer = LLMSummaryLayer()
        self.statistical_result = {
            "model_name": "StatisticalRiskModel",
            "overall_score": 45.0,
            "risk_level": "Moderate",
            "factor_breakdown": [
                {"factor": "annualized_volatility", "raw_value": 0.25, "normalized_score": 50.0, "weight": 15},
                {"factor": "maximum_drawdown", "raw_value": 0.30, "normalized_score": 40.0, "weight": 15},
            ],
            "raw_metrics": {"annualized_volatility": 0.25, "maximum_drawdown": 0.30},
            "weights_used": {"annualized_volatility": 15, "maximum_drawdown": 15},
            "explanation": "Sample explanation.",
        }
        self.structural_result = {
            "model_name": "StructuralRiskModel",
            "overall_score": 35.0,
            "risk_level": "Moderate",
            "factor_breakdown": [
                {"factor": "profitability_risk", "raw_value": 0.2, "normalized_score": 40.0, "weight": 15},
                {"factor": "leverage_risk", "raw_value": 0.3, "normalized_score": 30.0, "weight": 15},
            ],
            "raw_metrics": {"profitability_risk": 0.2, "leverage_risk": 0.3},
            "weights_used": {"profitability_risk": 15, "leverage_risk": 15},
            "missing_data_notes": [],
            "explanation": "Sample structural explanation.",
        }

    def test_generate_summary_returns_expected_structure(self):
        result = self.layer.generate_summary(
            ticker="AAPL",
            statistical_result=self.statistical_result,
            structural_result=self.structural_result,
        )

        self.assertIsInstance(result, LLMSummaryOutput)
        self.assertIsInstance(result.short_summary, str)
        self.assertIsInstance(result.long_summary, str)
        self.assertIsInstance(result.key_risk_drivers, list)
        self.assertIsInstance(result.model_disagreement_note, str)
        self.assertIsInstance(result.reserved_factor_note, str)
        self.assertIsInstance(result.final_combined_interpretation, str)

    def test_generate_summary_with_mock_llm(self):
        result = self.layer.generate_summary(
            ticker="AAPL",
            statistical_result=self.statistical_result,
            structural_result=self.structural_result,
        )

        self.assertIn("AAPL", result.short_summary)
        self.assertIn("45.0", result.short_summary)
        self.assertIn("35.0", result.short_summary)
        self.assertIn("Models agree", result.model_disagreement_note)
        self.assertIn("placeholder", result.reserved_factor_note.lower())

    def test_generate_summary_with_custom_llm_provider(self):
        def custom_llm(prompt):
            return '{"short_summary": "Custom short.", "long_summary": "Custom long.", "key_risk_drivers": ["Custom"], "model_disagreement_note": "Custom note.", "reserved_factor_note": "Custom reserved.", "final_combined_interpretation": "Custom final."}'

        layer = LLMSummaryLayer(llm_provider=custom_llm)
        result = layer.generate_summary(
            ticker="TSLA",
            statistical_result=self.statistical_result,
            structural_result=self.structural_result,
        )

        self.assertEqual(result.short_summary, "Custom short.")
        self.assertEqual(result.long_summary, "Custom long.")
        self.assertTrue(len(result.key_risk_drivers) >= 1)

    def test_llm_summary_grounding_filters_unknown_drivers(self):
        def ungrounded_llm(prompt):
            return '{"short_summary": "Ungrounded", "long_summary": "Ungrounded", "key_risk_drivers": ["MadeUpRisk", "AnotherFakeRisk"], "model_disagreement_note": "note", "reserved_factor_note": "", "final_combined_interpretation": "interp"}'

        layer = LLMSummaryLayer(llm_provider=ungrounded_llm)
        result = layer.generate_summary(
            ticker="AAPL",
            statistical_result=self.statistical_result,
            structural_result=self.structural_result,
        )

        allowed = {
            "annualized_volatility",
            "maximum_drawdown",
            "profitability_risk",
            "leverage_risk",
        }
        self.assertTrue(all(driver in allowed for driver in result.key_risk_drivers))
        self.assertIn("AAA", result.reserved_factor_note)

    def test_parse_llm_response_handles_invalid_json(self):
        def invalid_llm(prompt):
            return "Invalid JSON"

        layer = LLMSummaryLayer(llm_provider=invalid_llm)
        result = layer.generate_summary(
            ticker="TEST",
            statistical_result=self.statistical_result,
            structural_result=self.structural_result,
        )

        self.assertEqual(result.short_summary, "Summary generation failed.")
        self.assertEqual(result.long_summary, "Unable to parse LLM response.")


if __name__ == "__main__":
    unittest.main()
