from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from config import ConfigLoader
from data.yfinance_fundamentals_adapter import YFinanceFundamentalsAdapter
from llm import LLMSummaryLayer, LLMProviderRunResult
from models import StatisticalRiskModel, StructuralRiskModel
from schemas import FactorConfig, ModelConfig, RESERVED_FACTORS
from schemas.output_schema import build_agent_output
from statistical_risk_model import YFinanceDataAdapter
from ui.ascii_chart import render_price_chart
from ui.chart_service import get_chart_time_series


LOGGER = logging.getLogger(__name__)


def _load_json_file(file_path: str) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _model_config_from_dict(config_data: Optional[Mapping[str, Any]]) -> Optional[ModelConfig]:
    if not config_data:
        return None

    threshold_map = config_data.get("factor_thresholds", config_data)
    parsed: Dict[str, FactorConfig] = {}
    for factor, cfg in threshold_map.items():
        if isinstance(cfg, FactorConfig):
            parsed[factor] = cfg
        else:
            parsed[factor] = FactorConfig(
                weight=float(cfg.get("weight", 0.0)),
                min_value=float(cfg.get("min_value", 0.0)),
                max_value=float(cfg.get("max_value", 1.0)),
                direction=cfg.get("direction", "higher_worse"),
            )

    # Ensure reserved placeholders are always present in runtime config.
    for reserved_factor in RESERVED_FACTORS:
        parsed.setdefault(
            reserved_factor,
            FactorConfig(weight=0.0, min_value=0.0, max_value=1.0, direction="higher_worse"),
        )

    return ModelConfig(factor_thresholds=parsed)


def _default_runtime_data() -> Dict[str, Any]:
    return {
        "price_history": [
            {"adj_close": 150.0, "volume": 1000000},
            {"adj_close": 151.5, "volume": 1010000},
            {"adj_close": 149.0, "volume": 980000},
            {"adj_close": 152.0, "volume": 1020000},
            {"adj_close": 153.0, "volume": 1030000},
            {"adj_close": 151.0, "volume": 990000},
        ],
        "benchmark_history": [
            {"adj_close": 420.0, "volume": 3000000},
            {"adj_close": 421.0, "volume": 3050000},
            {"adj_close": 419.0, "volume": 2990000},
            {"adj_close": 423.0, "volume": 3060000},
            {"adj_close": 425.0, "volume": 3080000},
            {"adj_close": 426.0, "volume": 3070000},
        ],
        "financial_data": {
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
        },
        "company_profile": {
            "sector": "Technology",
            "industry": "Software",
            "customer_concentration": 0.15,
            "product_concentration": 0.2,
        },
        "industry_metadata": {"sector_risk_multiplier": {"Technology": 1.0}},
    }


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _market_fetch_debug_stats(series: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    rows = series or []
    if not rows:
        return {
            "rows": 0,
            "date_start": None,
            "date_end": None,
            "first_price": None,
            "last_price": None,
        }

    first = rows[0]
    last = rows[-1]
    first_price = _to_float(first.get("adj_close"))
    if first_price is None:
        first_price = _to_float(first.get("close"))
    last_price = _to_float(last.get("adj_close"))
    if last_price is None:
        last_price = _to_float(last.get("close"))

    return {
        "rows": len(rows),
        "date_start": first.get("date"),
        "date_end": last.get("date"),
        "first_price": first_price,
        "last_price": last_price,
    }


def _load_market_data_with_fallback(ticker: str, benchmark: str, lookback: int) -> Tuple[Dict[str, Any], str, List[str], Dict[str, Any]]:
    warnings: List[str] = []
    fallback = _default_runtime_data()
    fallback_market = {
        "asset": fallback.get("price_history", []),
        "benchmark": fallback.get("benchmark_history", []),
    }

    adapter = YFinanceDataAdapter(fallback_data=fallback_market, logger=LOGGER)
    fetched = adapter.get_price_history(ticker=ticker, lookback_window=lookback, benchmark_ticker=benchmark)

    asset_rows = list(fetched.get("asset", []))
    benchmark_rows = list(fetched.get("benchmark", []))
    used_fallback = asset_rows == list(fallback_market["asset"]) and benchmark_rows == list(fallback_market["benchmark"])

    source = "local_sample_fallback" if used_fallback else "yfinance_adapter"
    if used_fallback:
        warnings.append("Market data source fallback activated via YFinanceDataAdapter.")

    debug_stats = {
        "asset": _market_fetch_debug_stats(asset_rows),
        "benchmark": _market_fetch_debug_stats(benchmark_rows),
    }

    return {
        "price_history": asset_rows,
        "benchmark_history": benchmark_rows,
    }, source, warnings, debug_stats


def _load_structural_data_with_fallback(ticker: str) -> Tuple[Dict[str, Any], str, List[str]]:
    warnings: List[str] = []
    fallback = _default_runtime_data()
    fallback_structural = {
        "financial_data": fallback.get("financial_data"),
        "company_profile": fallback.get("company_profile"),
        "industry_metadata": fallback.get("industry_metadata"),
    }

    adapter = YFinanceFundamentalsAdapter(logger=LOGGER)
    try:
        fetched = adapter.get_company_data(ticker=ticker)
        source = "yfinance_fundamentals_adapter"
        return {
            "financial_data": fetched.get("financial_data"),
            "company_profile": fetched.get("company_profile"),
            "industry_metadata": fetched.get("industry_metadata"),
        }, source, warnings
    except Exception as exc:
        warnings.append(f"Structural data source fallback activated: {exc}")
        LOGGER.warning("YFinanceFundamentalsAdapter failed, using fallback structural data: %s", exc)
        return fallback_structural, "local_sample_fallback", warnings


def _extract_structural_input_fields(runtime_data: Dict[str, Any]) -> Dict[str, Any]:
    financial_data = runtime_data.get("financial_data") or {}
    income = financial_data.get("income_statement") or {}
    balance = financial_data.get("balance_sheet") or {}
    cash_flow = financial_data.get("cash_flow") or {}
    market = financial_data.get("market") or {}
    profile = runtime_data.get("company_profile") or {}
    metadata = runtime_data.get("industry_metadata") or {}

    sector = profile.get("sector")
    sector_map = metadata.get("sector_risk_multiplier") if isinstance(metadata, Mapping) else {}

    return {
        "income_statement": {
            "net_income": income.get("net_income"),
            "revenue": income.get("revenue"),
            "gross_margin": income.get("gross_margin"),
            "operating_margin": income.get("operating_margin"),
            "roe": income.get("roe"),
            "roa": income.get("roa"),
            "revenue_history": income.get("revenue_history"),
            "eps_history": income.get("eps_history"),
            "net_income_history": income.get("net_income_history"),
        },
        "balance_sheet": {
            "total_debt": balance.get("total_debt"),
            "total_equity": balance.get("total_equity"),
            "current_ratio": balance.get("current_ratio"),
            "current_assets": balance.get("current_assets"),
            "current_liabilities": balance.get("current_liabilities"),
            "interest_expense": balance.get("interest_expense"),
            "ebit": balance.get("ebit"),
            "operating_income": balance.get("operating_income"),
        },
        "cash_flow": {
            "operating_cash_flow": cash_flow.get("operating_cash_flow"),
            "cash_flow_from_operations": cash_flow.get("cash_flow_from_operations"),
            "free_cash_flow": cash_flow.get("free_cash_flow"),
            "cash_flow_from_ops_history": cash_flow.get("cash_flow_from_ops_history"),
            "free_cash_flow_history": cash_flow.get("free_cash_flow_history"),
        },
        "market": {
            "market_cap": market.get("market_cap"),
            "pe_ratio": market.get("pe_ratio"),
            "ps_ratio": market.get("ps_ratio"),
            "peg_ratio": market.get("peg_ratio"),
        },
        "company_profile": {
            "sector": sector,
            "industry": profile.get("industry"),
            "customer_concentration": profile.get("customer_concentration"),
            "product_concentration": profile.get("product_concentration"),
        },
        "financial_data": {
            "altman_z_score": financial_data.get("altman_z_score"),
        },
        "industry_metadata": {
            "sector_risk_multiplier_for_sector": (
                sector_map.get(sector) if isinstance(sector_map, Mapping) and sector is not None else None
            )
        },
    }


def _load_runtime_payload(
    config_path: Optional[str],
    ticker: str,
    benchmark: str,
    lookback: int,
    prefer_real_market_data: bool,
) -> Tuple[Dict[str, Any], Optional[ModelConfig], Optional[ModelConfig], Dict[str, Any]]:
    data_source_info: Dict[str, Any] = {
        "market_data_source": "unknown",
        "warnings": [],
        "structural_data_source": "local_sample_fallback",
        "market_fetch_debug": {
            "asset": {"rows": 0, "date_start": None, "date_end": None, "first_price": None, "last_price": None},
            "benchmark": {"rows": 0, "date_start": None, "date_end": None, "first_price": None, "last_price": None},
        },
    }
    if not config_path:
        runtime = _default_runtime_data()
        if prefer_real_market_data:
            market_data, source, warnings, market_debug = _load_market_data_with_fallback(ticker, benchmark, lookback)
            runtime["price_history"] = market_data["price_history"]
            runtime["benchmark_history"] = market_data["benchmark_history"]
            structural_data, structural_source, structural_warnings = _load_structural_data_with_fallback(ticker)
            runtime["financial_data"] = structural_data["financial_data"]
            runtime["company_profile"] = structural_data["company_profile"]
            runtime["industry_metadata"] = structural_data["industry_metadata"]
            data_source_info["market_data_source"] = source
            data_source_info["structural_data_source"] = structural_source
            data_source_info["warnings"] = warnings + structural_warnings
            data_source_info["market_fetch_debug"] = market_debug
        else:
            data_source_info["market_data_source"] = "local_sample_fallback"
            data_source_info["warnings"] = ["Real market data disabled; using local sample payload."]
            data_source_info["market_fetch_debug"] = {
                "asset": _market_fetch_debug_stats(runtime.get("price_history")),
                "benchmark": _market_fetch_debug_stats(runtime.get("benchmark_history")),
            }
        return runtime, None, None, data_source_info

    payload = _load_json_file(config_path)
    data_section = payload.get("data", payload)
    statistical_cfg = _model_config_from_dict(payload.get("statistical_config"))
    structural_cfg = _model_config_from_dict(payload.get("structural_config"))
    market_data_source = "config_file"
    structural_data_source = "config_file" if data_section.get("financial_data") else "local_sample_fallback"
    if prefer_real_market_data and not data_section.get("price_history"):
        market_data, source, warnings, market_debug = _load_market_data_with_fallback(ticker, benchmark, lookback)
        data_section = dict(data_section)
        data_section["price_history"] = market_data["price_history"]
        data_section["benchmark_history"] = market_data["benchmark_history"]
        market_data_source = source
        data_source_info["warnings"] = warnings
        data_source_info["market_fetch_debug"] = market_debug
    elif data_section.get("price_history") and data_section.get("benchmark_history"):
        data_source_info["market_fetch_debug"] = {
            "asset": _market_fetch_debug_stats(data_section.get("price_history")),
            "benchmark": _market_fetch_debug_stats(data_section.get("benchmark_history")),
        }
    if prefer_real_market_data and not data_section.get("financial_data"):
        structural_data, structural_source, structural_warnings = _load_structural_data_with_fallback(ticker)
        data_section = dict(data_section)
        data_section["financial_data"] = structural_data["financial_data"]
        data_section["company_profile"] = structural_data["company_profile"]
        data_section["industry_metadata"] = structural_data["industry_metadata"]
        structural_data_source = structural_source
        data_source_info["warnings"] = data_source_info["warnings"] + structural_warnings
    data_source_info["market_data_source"] = market_data_source
    data_source_info["structural_data_source"] = structural_data_source
    return {
        "price_history": data_section.get("price_history"),
        "benchmark_history": data_section.get("benchmark_history"),
        "financial_data": data_section.get("financial_data"),
        "company_profile": data_section.get("company_profile"),
        "industry_metadata": data_section.get("industry_metadata"),
    }, statistical_cfg, structural_cfg, data_source_info


def _threshold_bucket(raw: Optional[float], min_value: float, max_value: float) -> str:
    if raw is None:
        return "missing"
    if raw < min_value:
        return "below_min"
    if raw > max_value:
        return "above_max"
    return "in_range"


def _price_field_usage(rows: Optional[List[Dict[str, Any]]]) -> Dict[str, int]:
    usage = {"adj_close": 0, "close": 0, "price": 0, "missing_price": 0, "with_date": 0}
    if not rows:
        return usage
    for row in rows:
        if "adj_close" in row and row.get("adj_close") is not None:
            usage["adj_close"] += 1
        elif "close" in row and row.get("close") is not None:
            usage["close"] += 1
        elif "price" in row and row.get("price") is not None:
            usage["price"] += 1
        else:
            usage["missing_price"] += 1
        if row.get("date"):
            usage["with_date"] += 1
    return usage


def _date_alignment_diagnostics(
    asset_rows: Optional[List[Dict[str, Any]]], benchmark_rows: Optional[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    asset_dates = {row.get("date") for row in (asset_rows or []) if row.get("date")}
    benchmark_dates = {row.get("date") for row in (benchmark_rows or []) if row.get("date")}
    overlap = asset_dates.intersection(benchmark_dates)
    return {
        "asset_points": len(asset_rows or []),
        "benchmark_points": len(benchmark_rows or []),
        "asset_has_dates": len(asset_dates) > 0,
        "benchmark_has_dates": len(benchmark_dates) > 0,
        "overlap_points": len(overlap),
        "uses_length_zip_for_beta": not (asset_dates and benchmark_dates),
    }


def _build_model_factor_diagnostics(model_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    factor_rows: List[Dict[str, Any]] = []
    weights = model_output.get("weights", {})
    total_weight = sum(float(w) for w in weights.values() if float(w) > 0)
    for factor in model_output.get("factor_scores", []):
        raw_value = factor.get("raw_value")
        score = float(factor.get("score", 0.0))
        weight = float(factor.get("weight", 0.0))
        threshold = factor.get("thresholds", {})
        min_value = float(threshold.get("min_value", 0.0))
        max_value = float(threshold.get("max_value", 1.0))
        contribution = (score * weight / total_weight) if total_weight > 0 and weight > 0 else 0.0
        factor_rows.append(
            {
                "factor": factor.get("factor"),
                "raw_value": raw_value,
                "normalized_score": score,
                "weight": weight,
                "weighted_contribution_to_total": round(contribution, 4),
                "threshold_bucket": _threshold_bucket(raw_value, min_value, max_value),
                "thresholds": threshold,
            }
        )
    return factor_rows


def _missing_data_notes_for_structural(runtime_data: Dict[str, Any], struct_model: StructuralRiskModel) -> List[str]:
    notes: List[str] = []
    financial_data = runtime_data.get("financial_data") or {}
    company_profile = runtime_data.get("company_profile") or {}
    industry_metadata = runtime_data.get("industry_metadata")
    # Reuse current model internals for diagnostics without changing score path.
    struct_model._calculate_raw_metrics(financial_data, company_profile, industry_metadata, notes)
    return notes


def _low_score_explanation(debug_data: Dict[str, Any]) -> List[str]:
    explanations: List[str] = []
    stat_factors = debug_data["statistical"]["factor_diagnostics"]
    low_stat = [f for f in stat_factors if f["weight"] > 0 and f["weighted_contribution_to_total"] < 2.5]
    for row in low_stat[:5]:
        explanations.append(
            f"Statistical factor {row['factor']} contributes only {row['weighted_contribution_to_total']:.2f} points; "
            f"raw={row['raw_value']} bucket={row['threshold_bucket']} thresholds={row['thresholds']}."
        )

    structural_notes = debug_data["structural"].get("missing_data_notes", [])
    if structural_notes:
        explanations.append(
            "Structural model has missing-data notes; some components may be understated when sub-metrics are absent: "
            + "; ".join(structural_notes)
        )

    if debug_data["data_source"]["market_data_source"] != "yfinance_adapter":
        explanations.append(
            "Statistical score is based on fallback sample market data, not live TSLA history, which can materially understate risk."
        )

    return explanations


def _print_debug_diagnostics(debug_data: Dict[str, Any]) -> None:
    print("=")
    print("DEBUG DIAGNOSTICS")
    print("=")
    print("Data source:")
    print(json.dumps(debug_data["data_source"], indent=2))
    print(f"Structural data source: {debug_data['data_source'].get('structural_data_source')}")
    print("-")
    print("Price field handling:")
    print(json.dumps(debug_data["price_field_handling"], indent=2))
    print("-")
    print("Date alignment:")
    print(json.dumps(debug_data["date_alignment"], indent=2))
    print("-")
    print("Threshold and direction audit:")
    print(json.dumps(debug_data["threshold_audit"], indent=2))
    print("-")
    print("Weight audit:")
    print(json.dumps(debug_data["weight_audit"], indent=2))
    print("-")
    print("Statistical factor diagnostics:")
    print(json.dumps(debug_data["statistical"]["factor_diagnostics"], indent=2))
    print("-")
    print("Structural factor diagnostics:")
    print(json.dumps(debug_data["structural"]["factor_diagnostics"], indent=2))
    print("-")
    print("Structural missing-data notes:")
    print(json.dumps(debug_data["structural"]["missing_data_notes"], indent=2))
    print("-")
    print("Structural input fields used for scoring:")
    print(json.dumps(debug_data["structural"]["input_fields"], indent=2))
    print("-")
    print("TSLA low-score explanation:")
    print(json.dumps(debug_data["underestimation_explanation"], indent=2))


def run_stock_risk_agent(
    ticker: str,
    benchmark: str,
    lookback: int,
    disable_llm: bool,
    config_path: Optional[str] = None,
    debug: bool = False,
    prefer_real_market_data: bool = True,
    llm_run_all_providers: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    runtime_data, stat_cfg_override, struct_cfg_override, data_source_info = _load_runtime_payload(
        config_path,
        ticker=ticker,
        benchmark=benchmark,
        lookback=lookback,
        prefer_real_market_data=prefer_real_market_data,
    )

    stat_config = stat_cfg_override or ConfigLoader.load_default_statistical_config()
    struct_config = struct_cfg_override or ConfigLoader.load_default_structural_config()

    stat_model = StatisticalRiskModel(config=stat_config)
    stat_result = stat_model.evaluate(
        ticker=ticker,
        benchmark_ticker=benchmark,
        lookback_window=lookback,
        price_history=runtime_data.get("price_history"),
        benchmark_history=runtime_data.get("benchmark_history"),
    )

    struct_model = StructuralRiskModel(config=struct_config)
    struct_result = struct_model.evaluate(
        ticker=ticker,
        financial_data=runtime_data.get("financial_data"),
        company_profile=runtime_data.get("company_profile"),
        industry_metadata=runtime_data.get("industry_metadata"),
    )

    llm_summary = None
    llm_provider_used: Optional[str] = None
    llm_provider_attempts: List[str] = []
    llm_provider_errors: Dict[str, str] = {}
    llm_provider_results: Dict[str, Dict[str, Any]] = {}
    if not disable_llm:
        summary_layer = LLMSummaryLayer()
        if llm_run_all_providers:
            all_results = summary_layer.generate_all_provider_summaries(
                ticker=ticker,
                statistical_result=stat_result,
                structural_result=struct_result,
                reserved_factors_status={factor: "inactive_placeholder" for factor in RESERVED_FACTORS},
            )

            llm_provider_attempts = list(summary_layer.PROVIDER_ORDER)
            for name, result in all_results.items():
                item: Dict[str, Any] = {
                    "status": result.status,
                    "reason": result.reason,
                    "summary": result.summary.__dict__ if result.summary else None,
                }
                llm_provider_results[name] = item
                if result.status == "failed" and result.reason:
                    llm_provider_errors[name] = result.reason

            # Keep output schema single-summary by selecting first successful provider in order.
            chosen: Optional[LLMProviderRunResult] = None
            for name in summary_layer.PROVIDER_ORDER:
                result = all_results.get(name)
                if result and result.status == "success" and result.summary is not None:
                    chosen = result
                    break
            if chosen is not None:
                llm_summary = chosen.summary
                llm_provider_used = chosen.provider
        else:
            llm_summary = summary_layer.generate_summary(
                ticker=ticker,
                statistical_result=stat_result,
                structural_result=struct_result,
                reserved_factors_status={factor: "inactive_placeholder" for factor in RESERVED_FACTORS},
            )
            llm_provider_used = summary_layer.last_provider_used
            llm_provider_attempts = list(summary_layer.provider_attempts)
            llm_provider_errors = dict(summary_layer.provider_errors)

    output = build_agent_output(
        ticker=ticker,
        benchmark=benchmark,
        statistical_result=stat_result,
        structural_result=struct_result,
        llm_summary=llm_summary,
    )

    stat_factors = output["statistical_risk"]["factor_scores"]
    struct_factors = output["structural_risk"]["factor_scores"]

    stat_required_zero_weight = [
        f for f in stat_factors if f["factor"] not in RESERVED_FACTORS and float(f["weight"]) == 0.0
    ]
    struct_required_zero_weight = [
        f for f in struct_factors if f["factor"] not in RESERVED_FACTORS and float(f["weight"]) == 0.0
    ]

    direction_anomalies: List[Dict[str, Any]] = []
    for model_name, factors in (("statistical", stat_factors), ("structural", struct_factors)):
        for factor in factors:
            direction = factor.get("thresholds", {}).get("direction")
            if direction != "higher_worse" and factor.get("factor") not in RESERVED_FACTORS:
                direction_anomalies.append(
                    {
                        "model": model_name,
                        "factor": factor.get("factor"),
                        "direction": direction,
                        "note": "Review normalization direction to ensure higher risk maps to higher score.",
                    }
                )

    debug_data: Dict[str, Any] = {
        "data_source": data_source_info,
        "price_field_handling": {
            "asset": _price_field_usage(runtime_data.get("price_history")),
            "benchmark": _price_field_usage(runtime_data.get("benchmark_history")),
            "price_preference_order": ["adj_close", "adjclose", "close", "price"],
        },
        "date_alignment": _date_alignment_diagnostics(
            runtime_data.get("price_history"),
            runtime_data.get("benchmark_history"),
        ),
        "threshold_audit": {
            "statistical": {
                "factors": [{"factor": f["factor"], "thresholds": f["thresholds"]} for f in stat_factors],
                "direction_anomalies": [d for d in direction_anomalies if d["model"] == "statistical"],
            },
            "structural": {
                "factors": [{"factor": f["factor"], "thresholds": f["thresholds"]} for f in struct_factors],
                "direction_anomalies": [d for d in direction_anomalies if d["model"] == "structural"],
            },
            "implementation_checks": {
                "max_drawdown_formula": "peak_to_trough drawdown using running peak",
                "beta_formula": "cov(asset, benchmark)/var(benchmark) using paired daily returns",
                "historical_var_95": "absolute 5th percentile of daily returns",
                "historical_cvar_95": "absolute mean of tail returns at/under VaR threshold",
            },
        },
        "weight_audit": {
            "statistical_zero_weight_required_factors": [f["factor"] for f in stat_required_zero_weight],
            "structural_zero_weight_required_factors": [f["factor"] for f in struct_required_zero_weight],
            "reserved_factors_expected_zero": RESERVED_FACTORS,
        },
        "statistical": {
            "score": output["statistical_risk"]["score"],
            "factor_diagnostics": _build_model_factor_diagnostics(output["statistical_risk"]),
        },
        "structural": {
            "score": output["structural_risk"]["score"],
            "factor_diagnostics": _build_model_factor_diagnostics(output["structural_risk"]),
            "missing_data_notes": _missing_data_notes_for_structural(runtime_data, struct_model),
            "input_fields": _extract_structural_input_fields(runtime_data),
        },
        "llm": {
            "enabled": not disable_llm,
            "provider_used": llm_provider_used,
            "provider_attempts": llm_provider_attempts,
            "provider_errors": llm_provider_errors,
            "provider_results": llm_provider_results,
            "run_all_mode": llm_run_all_providers,
        },
    }
    debug_data["underestimation_explanation"] = _low_score_explanation(debug_data)

    if debug:
        _print_debug_diagnostics(debug_data)

    return output, debug_data


def _print_formatted_summary(
    output: Dict[str, Any],
    llm_enabled: bool,
    llm_provider_used: Optional[str] = None,
    llm_provider_attempts: Optional[List[str]] = None,
    llm_provider_errors: Optional[Dict[str, str]] = None,
    llm_provider_results: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    stat = output["statistical_risk"]
    struct = output["structural_risk"]

    print(f"Ticker: {output['ticker']} | Benchmark: {output['benchmark']}")
    print(f"Timestamp: {output['timestamp']}")
    print("-")
    print(f"Statistical Risk -> score: {stat['score']:.2f}, level: {stat['level']}")
    print(f"Structural Risk  -> score: {struct['score']:.2f}, level: {struct['level']}")

    if llm_enabled and output.get("llm_summary"):
        llm_summary = output["llm_summary"]
        print("-")
        print("LLM Summary:")
        if llm_provider_used:
            print(f"  Provider Used: {llm_provider_used}")
        if llm_provider_attempts:
            print(f"  Attempts: {' -> '.join(llm_provider_attempts)}")
        if llm_provider_errors:
            print("  Failure Reasons:")
            for provider_name, reason in llm_provider_errors.items():
                print(f"    - {provider_name}: {reason}")
        print(f"  Short: {llm_summary['short_summary']}")
        print(f"  Disagreement: {llm_summary['model_disagreement_note']}")

    if llm_enabled and llm_provider_results:
        print("-")
        print("LLM Results:")
        provider_order = ["gemini", "ollama3_local", "fallback_mock"]
        for provider_name in provider_order:
            details = llm_provider_results.get(provider_name, {"status": "skipped", "reason": "not run", "summary": None})
            print(f"[{provider_name}]")
            print(f"Status: {details.get('status', 'skipped')}")
            reason = details.get("reason")
            if reason:
                print(f"Reason: {reason}")
            summary = details.get("summary")
            if details.get("status") == "success" and isinstance(summary, dict):
                print(f"Short: {summary.get('short_summary', '')}")
                print(f"Long: {summary.get('long_summary', '')}")
                drivers = summary.get("key_risk_drivers", [])
                print(f"Key Drivers: {', '.join(drivers) if drivers else 'N/A'}")
                print(f"Disagreement: {summary.get('model_disagreement_note', '')}")
                print(f"Reserved Factors: {summary.get('reserved_factor_note', '')}")
            print()

    print("-")
    print("Reserved Factors:")
    for factor in RESERVED_FACTORS:
        item = output["reserved_factors_status"][factor]
        print(
            f"  {factor}: enabled={item['enabled']}, weight={item['weight']}, "
            f"normalized_score={item['normalized_score']}, affects_final_score={item['affects_final_score']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock Risk Agent CLI")
    parser.add_argument("--ticker", default="TSLA", help="Stock ticker symbol (default: TSLA)")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark ticker (default: SPY)")
    parser.add_argument("--lookback", type=int, default=252, help="Lookback window for statistical model")
    parser.add_argument("--output-json", help="Optional output JSON file path")
    parser.add_argument("--disable-llm", action="store_true", help="Disable LLM summary generation")
    parser.add_argument("--debug", action="store_true", help="Print full scoring diagnostics and audit details")
    parser.add_argument(
        "--no-real-market-data",
        action="store_true",
        help="Disable live Yahoo market data fetch and force local sample market data.",
    )
    parser.add_argument(
        "--config",
        help="Optional config JSON path containing data and/or model config overrides",
    )
    parser.add_argument(
        "--chart",
        action="store_true",
        help="Render a 30-day ASCII price chart after the risk summary.",
    )
    parser.add_argument(
        "--llm-all-providers",
        action="store_true",
        help="Run Gemini, Ollama3 local, and fallback mock independently and print all LLM results.",
    )

    args = parser.parse_args()

    output, _debug_data = run_stock_risk_agent(
        ticker=args.ticker,
        benchmark=args.benchmark,
        lookback=args.lookback,
        disable_llm=args.disable_llm,
        config_path=args.config,
        debug=args.debug,
        prefer_real_market_data=not args.no_real_market_data,
        llm_run_all_providers=args.llm_all_providers,
    )

    _print_formatted_summary(
        output,
        llm_enabled=not args.disable_llm,
        llm_provider_used=_debug_data.get("llm", {}).get("provider_used"),
        llm_provider_attempts=_debug_data.get("llm", {}).get("provider_attempts"),
        llm_provider_errors=_debug_data.get("llm", {}).get("provider_errors"),
        llm_provider_results=_debug_data.get("llm", {}).get("provider_results"),
    )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Exported JSON to {args.output_json}")

    if args.chart:
        chart_data = get_chart_time_series(ticker=args.ticker, lookback_days=30)
        if chart_data["success"]:
            prices = [pt["price"] for pt in chart_data["points"]]
            print()
            print(chart_data["title"])
            for line in chart_data.get("summary_lines", []):
                print(line)
            print()
            print(render_price_chart(prices, height=12, point_char="*"))
        else:
            print(f"\n[Chart] {chart_data['error']}")


if __name__ == "__main__":
    main()
