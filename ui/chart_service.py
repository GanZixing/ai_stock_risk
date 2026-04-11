from __future__ import annotations

from datetime import datetime, time
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from statistical_risk_model import RiskDataAdapter, YFinanceDataAdapter


LOGGER = logging.getLogger(__name__)


def get_chart_time_series(
    ticker: str,
    lookback_days: int = 30,
    data_adapter: Optional[RiskDataAdapter] = None,
    intraday_fetcher: Optional[Callable[[str], Dict[str, Any]]] = None,
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
    intraday = (intraday_fetcher or _fetch_intraday_session_extremes)(symbol)
    summary_lines = build_chart_summary_lines(stats=stats, intraday=intraday)

    return {
        "success": True,
        "ticker": symbol,
        "lookback_days": lookback_days,
        "title": f"{symbol} - Last {lookback_days} Trading Days",
        "error": None,
        "points": points,
        "stats": stats,
        "intraday": intraday,
        "summary_lines": summary_lines,
        "source": "yfinance_adapter",
    }


def build_chart_summary_lines(
    stats: Mapping[str, Optional[float]],
    intraday: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    latest = _fmt_price(stats.get("latest_price"))
    ret = _fmt_pct(stats.get("return_30d_pct"))
    high = _fmt_price(stats.get("high_30d"))
    low = _fmt_price(stats.get("low_30d"))

    lines = [
        f"Latest: {latest}   Return: {ret}",
        f"30D High: {high}",
        f"30D Low : {low}",
    ]

    if intraday and intraday.get("has_intraday") and intraday.get("market_in_progress"):
        lines.append("")
        lines.append("Intraday Session:")
        lines.append(f"Premarket High: {_fmt_price(intraday.get('premarket_high'))}")
        lines.append(f"Premarket Low : {_fmt_price(intraday.get('premarket_low'))}")
        lines.append(f"Regular High  : {_fmt_price(intraday.get('regular_high'))}")
        lines.append(f"Regular Low   : {_fmt_price(intraday.get('regular_low'))}")
        lines.append(f"After-hours High: {_fmt_price(intraday.get('after_hours_high'))}")
        lines.append(f"After-hours Low : {_fmt_price(intraday.get('after_hours_low'))}")
    elif intraday and intraday.get("has_intraday") is False:
        lines.append("")
        lines.append(f"Intraday Session: Unavailable ({intraday.get('reason', 'N/A')})")

    return lines


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


def _fetch_intraday_session_extremes(ticker: str) -> Dict[str, Any]:
    try:
        import yfinance as yf
    except ImportError:
        return {"has_intraday": False, "reason": "yfinance not installed", "market_in_progress": False}

    try:
        frame = yf.download(
            tickers=ticker,
            period="1d",
            interval="5m",
            auto_adjust=False,
            prepost=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        return {"has_intraday": False, "reason": str(exc), "market_in_progress": False}

    if getattr(frame, "empty", True):
        return {"has_intraday": False, "reason": "no intraday data", "market_in_progress": False}

    if hasattr(frame.columns, "nlevels") and frame.columns.nlevels > 1:
        frame = frame.copy()
        frame.columns = frame.columns.get_level_values(0)

    now_et = datetime.now(ZoneInfo("America/New_York"))
    market_in_progress = now_et.weekday() < 5 and now_et.time() < time(16, 0)

    rows = []
    for idx, row in frame.iterrows():
        if not hasattr(idx, "astimezone"):
            continue
        ts_et = idx.astimezone(ZoneInfo("America/New_York"))
        if ts_et.date() != now_et.date():
            continue

        high_val = _to_float(row.get("High"))
        low_val = _to_float(row.get("Low"))
        if high_val is None and low_val is None:
            px = _to_float(row.get("Adj Close"))
            if px is None:
                px = _to_float(row.get("Close"))
            high_val = px
            low_val = px

        rows.append({"ts": ts_et, "high": high_val, "low": low_val})

    if not rows:
        return {"has_intraday": False, "reason": "today intraday not available", "market_in_progress": market_in_progress}

    pre_rows = [r for r in rows if time(4, 0) <= r["ts"].time() < time(9, 30)]
    reg_rows = [r for r in rows if time(9, 30) <= r["ts"].time() < time(16, 0)]
    aft_rows = [r for r in rows if time(16, 0) <= r["ts"].time() <= time(20, 0)]

    return {
        "has_intraday": True,
        "market_in_progress": market_in_progress,
        "premarket_high": _extreme(pre_rows, "high", max),
        "premarket_low": _extreme(pre_rows, "low", min),
        "regular_high": _extreme(reg_rows, "high", max),
        "regular_low": _extreme(reg_rows, "low", min),
        "after_hours_high": _extreme(aft_rows, "high", max),
        "after_hours_low": _extreme(aft_rows, "low", min),
    }


def _extreme(rows: Sequence[Mapping[str, Any]], key: str, fn: Callable[[Sequence[float]], float]) -> Optional[float]:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return fn(vals)


def _fmt_price(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"