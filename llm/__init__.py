from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from schemas import ModelResult

logger = logging.getLogger(__name__)


@dataclass
class LLMSummaryOutput:
    short_summary: str
    long_summary: str
    key_risk_drivers: List[str]
    model_disagreement_note: str
    reserved_factor_note: str
    final_combined_interpretation: str


class LLMSummaryLayer:
    def __init__(
        self,
        llm_provider: Optional[Callable[[str], str]] = None,
        prompt_template: Optional[str] = None,
    ):
        self.llm_provider = llm_provider or self._cascade_provider
        self.prompt_template = prompt_template or self._default_prompt_template()

    def generate_summary(
        self,
        ticker: str,
        statistical_result: Union[ModelResult, Dict[str, Any]],
        structural_result: Union[ModelResult, Dict[str, Any]],
        reserved_factors_status: Optional[Dict[str, Any]] = None,
        combined_metadata: Optional[Dict[str, Any]] = None,
    ) -> LLMSummaryOutput:
        stat_dict = statistical_result.to_dict() if hasattr(statistical_result, 'to_dict') else statistical_result
        struct_dict = structural_result.to_dict() if hasattr(structural_result, 'to_dict') else structural_result

        # Prepare the prompt
        prompt = self._build_prompt(
            ticker, stat_dict, struct_dict, reserved_factors_status, combined_metadata
        )

        # Call LLM (or mock)
        llm_response = self.llm_provider(prompt)

        # Parse the response into structured output
        parsed = self._parse_llm_response(llm_response)
        return self._enforce_grounding(parsed, stat_dict, struct_dict)

    def _enforce_grounding(
        self,
        summary: LLMSummaryOutput,
        statistical: Dict[str, Any],
        structural: Dict[str, Any],
    ) -> LLMSummaryOutput:
        allowed_drivers = self._allowed_driver_names(statistical, structural)
        allowed_lookup = {self._canonical_name(name): name for name in allowed_drivers}

        grounded_drivers: List[str] = []
        for driver in summary.key_risk_drivers:
            canonical = self._canonical_name(driver)
            if canonical in allowed_lookup:
                grounded_drivers.append(allowed_lookup[canonical])

        if not grounded_drivers:
            grounded_drivers = self._top_drivers_from_outputs(statistical, structural)

        reserved_note = summary.reserved_factor_note
        if not reserved_note:
            reserved_note = "AAA, BBB, and CCC are reserved placeholder factors with zero impact on the scores."

        return LLMSummaryOutput(
            short_summary=summary.short_summary,
            long_summary=summary.long_summary,
            key_risk_drivers=grounded_drivers,
            model_disagreement_note=summary.model_disagreement_note,
            reserved_factor_note=reserved_note,
            final_combined_interpretation=summary.final_combined_interpretation,
        )

    @staticmethod
    def _canonical_name(text: str) -> str:
        return text.replace("_", " ").strip().lower()

    def _allowed_driver_names(self, statistical: Dict[str, Any], structural: Dict[str, Any]) -> List[str]:
        names: List[str] = []
        for model_dict in (statistical, structural):
            for factor in model_dict.get("factor_breakdown", []):
                factor_name = factor.get("factor")
                if isinstance(factor_name, str):
                    names.append(factor_name)
            for factor in model_dict.get("factor_scores", []):
                factor_name = factor.get("factor")
                if isinstance(factor_name, str):
                    names.append(factor_name)
        return names

    @staticmethod
    def _top_drivers_from_outputs(statistical: Dict[str, Any], structural: Dict[str, Any]) -> List[str]:
        scored_factors: List[tuple[str, float]] = []
        for model_dict in (statistical, structural):
            for factor in model_dict.get("factor_breakdown", []):
                factor_name = factor.get("factor")
                normalized = factor.get("normalized_score", 0.0)
                if isinstance(factor_name, str):
                    scored_factors.append((factor_name, float(normalized)))
            for factor in model_dict.get("factor_scores", []):
                factor_name = factor.get("factor")
                normalized = factor.get("score", 0.0)
                if isinstance(factor_name, str):
                    scored_factors.append((factor_name, float(normalized)))

        scored_factors.sort(key=lambda item: item[1], reverse=True)
        deduped: List[str] = []
        seen = set()
        for name, _score in scored_factors:
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(name)
            if len(deduped) == 3:
                break

        return deduped

    def _default_prompt_template(self) -> str:
        return """
You are an expert financial analyst. Based on the provided model outputs, generate a summary explanation.

Inputs:
- Ticker: {ticker}
- Statistical Risk Model Output: {statistical_json}
- Structural Risk Model Output: {structural_json}
- Reserved Factors Status: {reserved_status}
- Combined Metadata: {metadata}

Requirements:
1. Summarize the Statistical Risk score.
2. Summarize the Structural Risk score.
3. Explain major contributing factors.
4. Explain if the two models disagree.
5. Explicitly mention that AAA, BBB, CCC are reserved placeholder factors and currently have zero impact.
6. Generate concise and extended summary versions.

Output format (JSON-like):
{{
    "short_summary": "Brief summary here.",
    "long_summary": "Detailed summary here.",
    "key_risk_drivers": ["Factor 1", "Factor 2"],
    "model_disagreement_note": "Note on disagreement.",
    "reserved_factor_note": "Note on placeholders.",
    "final_combined_interpretation": "Overall interpretation."
}}

Ensure the summary is grounded strictly in the provided model outputs. Do not invent or modify any scores or metrics.
"""

    def _build_prompt(
        self,
        ticker: str,
        statistical: Union[ModelResult, Dict[str, Any]],
        structural: Union[ModelResult, Dict[str, Any]],
        reserved: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]],
    ) -> str:
        import json
        stat_dict = statistical.to_dict() if hasattr(statistical, 'to_dict') else statistical
        struct_dict = structural.to_dict() if hasattr(structural, 'to_dict') else structural
        return self.prompt_template.format(
            ticker=ticker,
            statistical_json=json.dumps(stat_dict, indent=2),
            structural_json=json.dumps(struct_dict, indent=2),
            reserved_status=json.dumps(reserved or {}, indent=2),
            metadata=json.dumps(metadata or {}, indent=2),
        )

    @staticmethod
    def _gemini_provider(prompt: str) -> str:
        """Call Google Gemini API. Requires GEMINI_API_KEY environment variable."""
        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise RuntimeError("google-genai package is not installed") from exc

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text

    @staticmethod
    def _ollama_provider(prompt: str) -> str:
        """Call local Ollama API (llama3). Auto-detects availability on localhost:11434."""
        import json
        import urllib.request

        url = "http://localhost:11434/api/generate"
        payload = json.dumps({"model": "llama3", "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        return data["response"]

    @classmethod
    def _cascade_provider(cls, prompt: str) -> str:
        """Try Gemini → Ollama → mock, returning the first successful response."""
        import urllib.error

        providers = [
            ("Gemini", cls._gemini_provider),
            ("Ollama (llama3)", cls._ollama_provider),
            ("Mock", cls._mock_llm_provider),
        ]
        last_error: Optional[Exception] = None
        for name, provider in providers:
            try:
                result = provider(prompt)
                if name != "Mock":
                    logger.info("LLM cascade: using %s", name)
                return result
            except (RuntimeError, OSError, ImportError, urllib.error.URLError) as exc:
                logger.debug("LLM cascade: %s failed – %s", name, exc)
                last_error = exc
        # Should never reach here because _mock_llm_provider does not raise
        raise RuntimeError("All LLM providers failed") from last_error

    @staticmethod
    def _mock_llm_provider(prompt: str) -> str:
        # Mock implementation: parse the prompt and generate a basic response
        # In a real setup, this would call an actual LLM API
        import json
        import re

        # Extract ticker and scores from prompt (simplified)
        ticker_match = re.search(r'Ticker: (\w+)', prompt)
        ticker = ticker_match.group(1) if ticker_match else "UNKNOWN"

        stat_score_match = re.search(r'"overall_score": (\d+\.?\d*)', prompt)
        stat_score = float(stat_score_match.group(1)) if stat_score_match else 0.0

        struct_score_match = re.search(r'"overall_score": (\d+\.?\d*)', prompt.split('Structural Risk Model Output:')[1])
        struct_score = float(struct_score_match.group(1)) if struct_score_match else 0.0

        # Generate mock summaries
        short_summary = f"{ticker} has a statistical risk score of {stat_score} and structural risk score of {struct_score}."

        long_summary = f"For {ticker}, the Statistical Risk Model indicates a risk level of {'Low' if stat_score <= 25 else 'Moderate' if stat_score <= 50 else 'High' if stat_score <= 75 else 'Extreme'}, with key factors including volatility and drawdown. The Structural Risk Model shows a risk level of {'Low' if struct_score <= 25 else 'Moderate' if struct_score <= 50 else 'High' if struct_score <= 75 else 'Extreme'}, focusing on profitability and leverage. Major contributors are annualized volatility and profitability risk."

        disagreement = "Models agree." if abs(stat_score - struct_score) <= 10 else "Models disagree significantly."

        key_drivers = ["Annualized Volatility", "Profitability Risk", "Maximum Drawdown"]

        reserved_note = "AAA, BBB, and CCC are reserved placeholder factors with zero impact on the scores."

        combined = f"Overall, {ticker} exhibits {'low' if (stat_score + struct_score)/2 <= 25 else 'moderate' if (stat_score + struct_score)/2 <= 50 else 'high' if (stat_score + struct_score)/2 <= 75 else 'extreme'} combined risk."

        return json.dumps({
            "short_summary": short_summary,
            "long_summary": long_summary,
            "key_risk_drivers": key_drivers,
            "model_disagreement_note": disagreement,
            "reserved_factor_note": reserved_note,
            "final_combined_interpretation": combined,
        })

    @staticmethod
    def _parse_llm_response(response: str) -> LLMSummaryOutput:
        import json
        try:
            data = json.loads(response)
            return LLMSummaryOutput(
                short_summary=data.get("short_summary", ""),
                long_summary=data.get("long_summary", ""),
                key_risk_drivers=data.get("key_risk_drivers", []),
                model_disagreement_note=data.get("model_disagreement_note", ""),
                reserved_factor_note=data.get("reserved_factor_note", ""),
                final_combined_interpretation=data.get("final_combined_interpretation", ""),
            )
        except json.JSONDecodeError:
            # Fallback if parsing fails
            return LLMSummaryOutput(
                short_summary="Summary generation failed.",
                long_summary="Unable to parse LLM response.",
                key_risk_drivers=[],
                model_disagreement_note="",
                reserved_factor_note="",
                final_combined_interpretation="",
            )
