import unittest

from llm import LLMSummaryLayer, LLMSummaryOutput


class TestLLMSummaryLayer(unittest.TestCase):
    def setUp(self):
        self.layer = LLMSummaryLayer(llm_provider=LLMSummaryLayer._mock_llm_provider)
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

        self.assertIn("TEST", result.short_summary)
        self.assertIn("statistical risk score", result.short_summary.lower())

    def test_llm_failover_order_gemini_then_ollama_then_mock(self):
        layer = LLMSummaryLayer()

        calls = []

        def gemini_fail(_prompt):
            calls.append("gemini")
            raise RuntimeError("gemini down")

        def ollama_success(_prompt):
            calls.append("ollama")
            return (
                '{"short_summary":"ok","long_summary":"ok","key_risk_drivers":["annualized_volatility"],'
                '"model_disagreement_note":"note","reserved_factor_note":"note","final_combined_interpretation":"ok"}'
            )

        layer._generate_with_gemini = gemini_fail
        layer._generate_with_ollama3 = ollama_success

        result = layer.generate_summary(
            ticker="AAPL",
            statistical_result=self.statistical_result,
            structural_result=self.structural_result,
        )

        self.assertEqual(calls, ["gemini", "ollama"])
        self.assertEqual(result.short_summary, "ok")
        self.assertEqual(layer.last_provider_used, "ollama3_local")
        self.assertEqual(layer.provider_attempts, ["gemini", "ollama3_local"])

    def test_llm_failover_reaches_mock_when_both_external_fail(self):
        layer = LLMSummaryLayer()

        calls = []

        def gemini_fail(_prompt):
            calls.append("gemini")
            raise RuntimeError("gemini down")

        def ollama_fail(_prompt):
            calls.append("ollama")
            raise RuntimeError("ollama down")

        layer._generate_with_gemini = gemini_fail
        layer._generate_with_ollama3 = ollama_fail

        result = layer.generate_summary(
            ticker="AAPL",
            statistical_result=self.statistical_result,
            structural_result=self.structural_result,
        )

        self.assertEqual(calls, ["gemini", "ollama"])
        self.assertIn("AAPL", result.short_summary)
        self.assertEqual(layer.last_provider_used, "fallback_mock")
        self.assertEqual(layer.provider_attempts, ["gemini", "ollama3_local", "fallback_mock"])

    def test_llm_custom_provider_tracking(self):
        def custom_llm(_prompt):
            return '{"short_summary":"x","long_summary":"x","key_risk_drivers":["annualized_volatility"],"model_disagreement_note":"x","reserved_factor_note":"x","final_combined_interpretation":"x"}'

        layer = LLMSummaryLayer(llm_provider=custom_llm)
        _result = layer.generate_summary(
            ticker="AAPL",
            statistical_result=self.statistical_result,
            structural_result=self.structural_result,
        )
        self.assertEqual(layer.last_provider_used, "custom_provider")
        self.assertEqual(layer.provider_attempts, ["custom_provider"])

    def test_invalid_external_json_triggers_fallback_mock(self):
        layer = LLMSummaryLayer()

        def gemini_fail(_prompt):
            raise RuntimeError("gemini down")

        def ollama_invalid_json(_prompt):
            return "not-json-response"

        layer._generate_with_gemini = gemini_fail
        layer._generate_with_ollama3 = ollama_invalid_json

        result = layer.generate_summary(
            ticker="AAPL",
            statistical_result=self.statistical_result,
            structural_result=self.structural_result,
        )

        self.assertEqual(layer.last_provider_used, "fallback_mock")
        self.assertIn("AAPL", result.short_summary)

    def test_generate_all_provider_summaries_reports_statuses(self):
        layer = LLMSummaryLayer()

        def gemini_fail(_prompt):
            raise RuntimeError("GEMINI_API_KEY is not set")

        def ollama_ok(_prompt):
            return '{"short_summary":"ok","long_summary":"long","key_risk_drivers":["annualized_volatility"],"model_disagreement_note":"d","reserved_factor_note":"r","final_combined_interpretation":"f"}'

        layer._generate_with_gemini = gemini_fail
        layer._generate_with_ollama3 = ollama_ok

        results = layer.generate_all_provider_summaries(
            ticker="AAPL",
            statistical_result=self.statistical_result,
            structural_result=self.structural_result,
        )

        self.assertEqual(results["gemini"].status, "failed")
        self.assertIn("GEMINI_API_KEY", results["gemini"].reason)
        self.assertEqual(results["ollama3_local"].status, "success")
        self.assertIsNotNone(results["ollama3_local"].summary)
        self.assertEqual(results["fallback_mock"].status, "success")

    def test_generate_all_provider_summaries_marks_invalid_json_failed(self):
        layer = LLMSummaryLayer()

        def gemini_invalid(_prompt):
            return "not-json"

        def ollama_invalid(_prompt):
            return "still-not-json"

        layer._generate_with_gemini = gemini_invalid
        layer._generate_with_ollama3 = ollama_invalid

        results = layer.generate_all_provider_summaries(
            ticker="AAPL",
            statistical_result=self.statistical_result,
            structural_result=self.structural_result,
        )

        self.assertEqual(results["gemini"].status, "failed")
        self.assertIn("invalid JSON summary output", results["gemini"].reason)
        self.assertEqual(results["ollama3_local"].status, "failed")
        self.assertIn("invalid JSON summary output", results["ollama3_local"].reason)


if __name__ == "__main__":
    unittest.main()
