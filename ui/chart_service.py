from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

from statistical_risk_model import RiskDataAdapter, YFinanceDataAdapter


LOGGER = logging.getLogger(__name__)


def get_chart_time_series(
    ticker: str,
    lookback_days: int = 30,
    data_adapter: Optional[RiskDataAdapter] = None,
) -> Dict[str, Any]:
    """Return chart-ready price series and summary stats for a ticker.

    The function intentionally does not return synthetic fallback values. If live
    fetch fails, it returns success=False with a clear error message.
    """
    symbol = (ticker or "").strip().upper()
    if not symbol:
        return {
            "success": False,
            "ticker": symbol,
            "lookback_days": lookback_days,
            "title": "Invalid ticker",
            "error": "Ticker is required.",
            "points": [],
            "stats": None,
            "source": "none",
        }

    adapter = data_adapter or YFinanceDataAdapter(fallback_data=None, logger=LOGGER)
    try:
        # Reuse the existing adapter path without duplicating fetch logic.
        history = adapter.get_price_history(
            ticker=symbol,
            lookback_window=max(lookback_days, 30),
            benchmark_ticker=symbol,
        )
    except Exception as exc:
        return {
            "success": False,
            "ticker": symbol,
            "lookback_days": lookback_days,
            "title": f"{symbol} - Last {lookback_days} Trading Days",
            "error": f"Unable to fetch market data for {symbol}: {exc}",
            "points": [],
            "stats": None,
            "source": "fetch_failed",
        }

    rows = _normalize_rows(history.get("asset") or history.get("benchmark") or [])
    if not rows:
        return {
            "success": False,
            "ticker": symbol,
            "lookback_days": lookback_days,
            "title": f"{symbol} - Last {lookback_days} Trading Days",
            "error": f"No usable price rows returned for {symbol}.",
            "points": [],
            "stats": None,
            "source": "empty",
        }

    points = rows[-lookback_days:]
    stats = _build_stats(points)

    return {
        "success": True,
        "ticker": symbol,
        "lookback_days": lookback_days,
        "title": f"{symbol} - Last {lookback_days} Trading Days",
        "error": None,
        "points": points,
        "stats": stats,
        "source": "yfinance_adapter",
    }


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        date = row.get("date")
        if not date:
            continue
        price = _to_float(row.get("adj_close"))
        if price is None:
            price = _to_float(row.get("close"))
        if price is None:
            continue
        normalized.append({"date": str(date), "price": price})

    normalized.sort(key=lambda item: item["date"])
    return normalized


def _build_stats(points: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    if not points:
        return {
            "latest_price": None,
            "high_30d": None,
            "low_30d": None,
            "return_30d_pct": None,
        }

    prices = [float(point["price"]) for point in points]
    latest = prices[-1]
    high_value = max(prices)
    low_value = min(prices)

    first = prices[0]
    if first == 0:
        period_return = None
    else:
        period_return = ((latest - first) / abs(first)) * 100.0

    return {
        "latest_price": latest,
        "period_high": high_value,
        "period_low": low_value,
        "period_return_pct": period_return,
        "high_30d": high_value,
        "low_30d": low_value,
        "return_30d_pct": period_return,
    }