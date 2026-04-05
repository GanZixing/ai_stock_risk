import json
import unittest
from unittest.mock import MagicMock, patch

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

    # ------------------------------------------------------------------ #
    # Tests for _gemini_provider                                           #
    # ------------------------------------------------------------------ #

    def test_gemini_provider_raises_when_api_key_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError, msg="Should raise when GEMINI_API_KEY is absent"):
                LLMSummaryLayer._gemini_provider("test prompt")

    def test_gemini_provider_calls_api_and_returns_text(self):
        mock_response = MagicMock()
        mock_response.text = '{"short_summary": "Gemini summary.", "long_summary": "Long.", "key_risk_drivers": [], "model_disagreement_note": "", "reserved_factor_note": "", "final_combined_interpretation": ""}'

        mock_models = MagicMock()
        mock_models.generate_content.return_value = mock_response

        mock_client_instance = MagicMock()
        mock_client_instance.models = mock_models

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client_instance

        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
            with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai}):
                result = LLMSummaryLayer._gemini_provider("test prompt")

        self.assertEqual(result, mock_response.text)
        mock_genai.Client.assert_called_once_with(api_key="fake-key")

    # ------------------------------------------------------------------ #
    # Tests for _ollama_provider                                           #
    # ------------------------------------------------------------------ #

    def test_ollama_provider_returns_response_text(self):
        expected_text = '{"short_summary": "Ollama summary.", "long_summary": "Long.", "key_risk_drivers": [], "model_disagreement_note": "", "reserved_factor_note": "", "final_combined_interpretation": ""}'
        api_payload = json.dumps({"response": expected_text}).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = api_payload
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = LLMSummaryLayer._ollama_provider("test prompt")

        self.assertEqual(result, expected_text)

    def test_ollama_provider_raises_on_connection_error(self):
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
            with self.assertRaises(urllib.error.URLError):
                LLMSummaryLayer._ollama_provider("test prompt")

    # ------------------------------------------------------------------ #
    # Tests for _cascade_provider                                          #
    # ------------------------------------------------------------------ #

    def test_cascade_uses_gemini_when_available(self):
        expected = '{"short_summary": "Gemini.", "long_summary": "L.", "key_risk_drivers": [], "model_disagreement_note": "", "reserved_factor_note": "", "final_combined_interpretation": ""}'
        with patch.object(LLMSummaryLayer, "_gemini_provider", return_value=expected) as mock_gemini:
            result = LLMSummaryLayer._cascade_provider("prompt")
        self.assertEqual(result, expected)
        mock_gemini.assert_called_once_with("prompt")

    def test_cascade_falls_back_to_ollama_when_gemini_fails(self):
        expected = '{"short_summary": "Ollama.", "long_summary": "L.", "key_risk_drivers": [], "model_disagreement_note": "", "reserved_factor_note": "", "final_combined_interpretation": ""}'
        with patch.object(LLMSummaryLayer, "_gemini_provider", side_effect=RuntimeError("no key")):
            with patch.object(LLMSummaryLayer, "_ollama_provider", return_value=expected) as mock_ollama:
                result = LLMSummaryLayer._cascade_provider("prompt")
        self.assertEqual(result, expected)
        mock_ollama.assert_called_once_with("prompt")

    def test_cascade_falls_back_to_mock_when_gemini_and_ollama_fail(self):
        mock_json = '{"short_summary": "X", "long_summary": "Y", "key_risk_drivers": [], "model_disagreement_note": "", "reserved_factor_note": "", "final_combined_interpretation": ""}'
        with patch.object(LLMSummaryLayer, "_gemini_provider", side_effect=RuntimeError("no key")):
            with patch.object(LLMSummaryLayer, "_ollama_provider", side_effect=OSError("no ollama")):
                with patch.object(LLMSummaryLayer, "_mock_llm_provider", return_value=mock_json) as mock_mock:
                    result = LLMSummaryLayer._cascade_provider("prompt")
        self.assertEqual(result, mock_json)
        mock_mock.assert_called_once_with("prompt")

    def test_default_provider_is_cascade(self):
        layer = LLMSummaryLayer()
        # When no API key is set and Ollama is unavailable, the default cascade
        # provider falls back to the mock and still returns a valid JSON string.
        import json
        import urllib.error
        with patch.dict("os.environ", {}, clear=True):
            with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
                with patch.object(LLMSummaryLayer, "_mock_llm_provider", return_value='{"short_summary": "ok"}') as mock_fn:
                    raw = layer.llm_provider("dummy")
        parsed = json.loads(raw)
        self.assertIn("short_summary", parsed)
        mock_fn.assert_called_once_with("dummy")

    def test_generate_summary_uses_cascade_by_default(self):
        """End-to-end: with no API key and no Ollama, cascade falls back to mock."""
        import urllib.error
        with patch.dict("os.environ", {}, clear=False) as env:
            env.pop("GEMINI_API_KEY", None)
            with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
                layer = LLMSummaryLayer()
                result = layer.generate_summary(
                    ticker="AAPL",
                    statistical_result=self.statistical_result,
                    structural_result=self.structural_result,
                )
        self.assertIsInstance(result, LLMSummaryOutput)
        self.assertIn("AAPL", result.short_summary)


if __name__ == "__main__":
    unittest.main()
