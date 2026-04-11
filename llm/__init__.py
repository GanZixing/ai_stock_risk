from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from schemas import ModelResult


LOGGER = logging.getLogger(__name__)


@dataclass
class LLMSummaryOutput:
    short_summary: str
    long_summary: str
    key_risk_drivers: List[str]
    model_disagreement_note: str
    reserved_factor_note: str
    final_combined_interpretation: str


@dataclass
class LLMProviderRunResult:
    provider: str
    status: str  # success | failed | skipped
    reason: Optional[str]
    summary: Optional[LLMSummaryOutput]


class LLMSummaryLayer:
    PROVIDER_ORDER = ("gemini", "ollama3_local", "fallback_mock")

    def __init__(
        self,
        llm_provider: Optional[Callable[[str], str]] = None,
        prompt_template: Optional[str] = None,
    ):
        # Optional direct provider override for tests/custom integrations.
        self.llm_provider = llm_provider
        self.prompt_template = prompt_template or self._default_prompt_template()
        self.last_provider_used: str = "not_run"
        self.provider_attempts: List[str] = []
        self.provider_errors: Dict[str, str] = {}

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

        # Call LLM in order: Gemini -> local Ollama3 -> existing fallback mock.
        llm_response = self._generate_with_failover(prompt)

        # Parse the response into structured output
        parsed, parse_reason = self._parse_llm_response_with_reason(llm_response)
        if parsed.short_summary == "Summary generation failed." and self.last_provider_used != "fallback_mock":
            provider = self.last_provider_used
            if provider:
                self.provider_errors[provider] = parse_reason or "invalid JSON summary output"
            self.last_provider_used = "fallback_mock"
            if "fallback_mock" not in self.provider_attempts:
                self.provider_attempts.append("fallback_mock")
            parsed, _ = self._parse_llm_response_with_reason(self._mock_llm_provider(prompt))

        return self._enforce_grounding(parsed, stat_dict, struct_dict)

    def generate_all_provider_summaries(
        self,
        ticker: str,
        statistical_result: Union[ModelResult, Dict[str, Any]],
        structural_result: Union[ModelResult, Dict[str, Any]],
        reserved_factors_status: Optional[Dict[str, Any]] = None,
        combined_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, LLMProviderRunResult]:
        stat_dict = statistical_result.to_dict() if hasattr(statistical_result, "to_dict") else statistical_result
        struct_dict = structural_result.to_dict() if hasattr(structural_result, "to_dict") else structural_result
        prompt = self._build_prompt(
            ticker,
            stat_dict,
            struct_dict,
            reserved_factors_status,
            combined_metadata,
        )

        results: Dict[str, LLMProviderRunResult] = {}
        for provider_name in self.PROVIDER_ORDER:
            results[provider_name] = self._run_single_provider(provider_name, prompt, stat_dict, struct_dict)
        return results

    def _run_single_provider(
        self,
        provider_name: str,
        prompt: str,
        statistical: Dict[str, Any],
        structural: Dict[str, Any],
    ) -> LLMProviderRunResult:
        if provider_name == "gemini" and os.getenv("LLM_SKIP_GEMINI", "0") == "1":
            return LLMProviderRunResult(provider=provider_name, status="skipped", reason="disabled by LLM_SKIP_GEMINI=1", summary=None)
        if provider_name == "ollama3_local" and os.getenv("LLM_SKIP_OLLAMA", "0") == "1":
            return LLMProviderRunResult(provider=provider_name, status="skipped", reason="disabled by LLM_SKIP_OLLAMA=1", summary=None)

        if provider_name == "gemini":
            provider_fn = self._generate_with_gemini
        elif provider_name == "ollama3_local":
            provider_fn = self._generate_with_ollama3
        elif provider_name == "fallback_mock":
            provider_fn = self._mock_llm_provider
        else:
            return LLMProviderRunResult(provider=provider_name, status="failed", reason="unknown provider", summary=None)

        try:
            raw = provider_fn(prompt)
        except Exception as exc:
            return LLMProviderRunResult(provider=provider_name, status="failed", reason=str(exc), summary=None)

        parsed, parse_reason = self._parse_llm_response_with_reason(raw)
        if parsed.short_summary == "Summary generation failed.":
            return LLMProviderRunResult(
                provider=provider_name,
                status="failed",
                reason=parse_reason or "invalid JSON summary output",
                summary=None,
            )

        grounded = self._enforce_grounding(parsed, statistical, structural)
        return LLMProviderRunResult(provider=provider_name, status="success", reason=None, summary=grounded)

    def _generate_with_failover(self, prompt: str) -> str:
        self.provider_attempts = []
        self.provider_errors = {}

        if self.llm_provider is not None:
            self.provider_attempts.append("custom_provider")
            self.last_provider_used = "custom_provider"
            return self.llm_provider(prompt)

        last_error: Optional[Exception] = None

        providers = (
            ("gemini", self._generate_with_gemini),
            ("ollama3_local", self._generate_with_ollama3),
        )

        for provider_name, provider in providers:
            self.provider_attempts.append(provider_name)
            try:
                response = provider(prompt)
                if response:
                    self.last_provider_used = provider_name
                    return response
            except Exception as exc:
                last_error = exc
                self.provider_errors[provider_name] = str(exc)
                LOGGER.warning("LLM provider '%s' failed: %s", provider_name, exc)

        if last_error is not None:
            # Keep fallback deterministic and available even when remote APIs fail.
            self.last_provider_used = "fallback_mock"
            self.provider_attempts.append("fallback_mock")
            return self._mock_llm_provider(prompt)
        self.last_provider_used = "fallback_mock"
        self.provider_attempts.append("fallback_mock")
        return self._mock_llm_provider(prompt)

    def _generate_with_gemini(self, prompt: str) -> str:
        api_key = os.getenv("GEMINI_API_KEY") or self._read_local_env_value("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (env or .env.local)")

        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            f"?key={api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }
        req = urllib_request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=600) as response:
                body = response.read().decode("utf-8")
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Gemini request failed: {exc}") from exc

        parsed = json.loads(body)
        candidates = parsed.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no candidates")
        parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
        if not parts:
            raise RuntimeError("Gemini returned empty content parts")
        text = parts[0].get("text")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Gemini returned empty text")
        return text

    def _generate_with_ollama3(self, prompt: str) -> str:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3:latest")
        timeout_seconds = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600"))
        url = f"{base_url.rstrip('/')}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        req = urllib_request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            parsed_url = urllib_parse.urlparse(url)
            host = (parsed_url.hostname or "").lower()
            bypass_proxy = host in {"localhost", "127.0.0.1", "::1"}

            if bypass_proxy:
                opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))
                response_ctx = opener.open(req, timeout=timeout_seconds)
            else:
                response_ctx = urllib_request.urlopen(req, timeout=timeout_seconds)

            with response_ctx as response:
                body = response.read().decode("utf-8")
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        parsed = json.loads(body)
        text = parsed.get("response")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Ollama returned empty response text")
        return text

    @staticmethod
    def _read_local_env_value(key: str) -> Optional[str]:
        env_path = Path.cwd() / ".env.local"
        if not env_path.exists():
            return None
        try:
            with env_path.open("r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in stripped:
                        continue
                    env_key, env_val = stripped.split("=", 1)
                    if env_key.strip() == key:
                        return env_val.strip().strip('"').strip("'")
        except OSError:
            return None
        return None

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
        parsed, _ = LLMSummaryLayer._parse_llm_response_with_reason(response)
        return parsed

    @staticmethod
    def _parse_llm_response_with_reason(response: str) -> Tuple[LLMSummaryOutput, Optional[str]]:
        import json

        def _try_parse(text: str) -> LLMSummaryOutput:
            data = json.loads(text)
            return LLMSummaryOutput(
                short_summary=data.get("short_summary", ""),
                long_summary=data.get("long_summary", ""),
                key_risk_drivers=data.get("key_risk_drivers", []),
                model_disagreement_note=data.get("model_disagreement_note", ""),
                reserved_factor_note=data.get("reserved_factor_note", ""),
                final_combined_interpretation=data.get("final_combined_interpretation", ""),
            )

        # First attempt: parse as-is (covers clean JSON responses).
        try:
            return _try_parse(response), None
        except json.JSONDecodeError:
            pass

        # Second attempt: strip prose prefix/suffix by extracting the first {...} block.
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return _try_parse(response[start : end + 1]), None
            except json.JSONDecodeError:
                pass

        # Final failure: report with preview.
        try:
            json.loads(response)
        except json.JSONDecodeError as exc:
            preview = response.strip().replace("\n", " ").replace("\r", " ")[:180]
            reason = f"invalid JSON summary output: {exc.msg} (near pos {exc.pos}); preview='{preview}'"
        else:
            reason = "invalid JSON summary output"
        return LLMSummaryOutput(
            short_summary="Summary generation failed.",
            long_summary="Unable to parse LLM response.",
            key_risk_drivers=[],
            model_disagreement_note="",
            reserved_factor_note="",
            final_combined_interpretation="",
        ), reason
