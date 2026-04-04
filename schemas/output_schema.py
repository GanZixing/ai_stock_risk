from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from llm import LLMSummaryOutput
from schemas import ModelResult, RESERVED_FACTORS


def _default_model_notes() -> List[str]:
    return [
        "AAA/BBB/CCC are reserved inactive placeholder factors with zero default impact.",
    ]


def format_model_output(result: ModelResult, notes: Optional[List[str]] = None) -> Dict[str, Any]:
    factor_scores = [
        {
            "factor": factor.factor,
            "score": float(factor.normalized_score),
            "raw_value": factor.raw_value,
            "weight": float(factor.weight),
            "thresholds": factor.thresholds,
        }
        for factor in result.factor_breakdown
    ]

    return {
        "model_name": result.model_name,
        "score": float(result.overall_score),
        "level": result.risk_level,
        "factor_scores": factor_scores,
        "raw_metrics": result.raw_metrics,
        "weights": result.weights_used,
        "notes": notes if notes is not None else _default_model_notes(),
    }


def build_reserved_factors_status(
    statistical_result: ModelResult,
    structural_result: ModelResult,
) -> Dict[str, Dict[str, Any]]:
    status: Dict[str, Dict[str, Any]] = {}

    for factor in RESERVED_FACTORS:
        stat_status = statistical_result.reserved_factors.get(factor, {})
        struct_status = structural_result.reserved_factors.get(factor, {})

        stat_weight = float(stat_status.get("weight", statistical_result.weights_used.get(factor, 0.0)))
        struct_weight = float(struct_status.get("weight", structural_result.weights_used.get(factor, 0.0)))

        enabled = (stat_weight > 0) or (struct_weight > 0)
        weight = stat_weight if stat_weight > 0 else struct_weight

        raw_value = stat_status.get("raw_value")
        if raw_value is None:
            raw_value = struct_status.get("raw_value")

        normalized_score = float(stat_status.get("normalized_score", 0.0))
        if normalized_score == 0.0:
            normalized_score = float(struct_status.get("normalized_score", 0.0))

        status[factor] = {
            "enabled": bool(enabled),
            "weight": float(weight),
            "raw_value": raw_value,
            "normalized_score": float(normalized_score),
            "affects_final_score": bool(enabled and weight > 0),
        }

    return status


def format_llm_summary(summary: Optional[LLMSummaryOutput]) -> Optional[Dict[str, Any]]:
    if summary is None:
        return None

    return {
        "short_summary": summary.short_summary,
        "long_summary": summary.long_summary,
        "key_risk_drivers": summary.key_risk_drivers,
        "model_disagreement_note": summary.model_disagreement_note,
        "reserved_factor_note": summary.reserved_factor_note,
        "final_combined_interpretation": summary.final_combined_interpretation,
    }


def build_agent_output(
    ticker: str,
    benchmark: str,
    statistical_result: ModelResult,
    structural_result: ModelResult,
    llm_summary: Optional[LLMSummaryOutput] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    ts = timestamp or datetime.now(timezone.utc).isoformat()

    return {
        "ticker": ticker,
        "benchmark": benchmark,
        "timestamp": ts,
        "statistical_risk": format_model_output(statistical_result),
        "structural_risk": format_model_output(structural_result),
        "llm_summary": format_llm_summary(llm_summary),
        "reserved_factors_status": build_reserved_factors_status(statistical_result, structural_result),
    }
