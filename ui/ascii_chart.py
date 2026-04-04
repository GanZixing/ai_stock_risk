from __future__ import annotations

from typing import List, Optional, Sequence


_CHART_HEIGHT = 12  # inner grid rows (excluding axes)
_POINT_CHAR = "█"
_AXIS_V = "│"
_AXIS_H = "─"
_ORIGIN = "└"


def render_price_chart(
    prices: Sequence[float],
    title: str = "",
    height: int = _CHART_HEIGHT,
) -> str:
    """Return a multi-line ASCII price chart string.

    Parameters
    ----------
    prices:
        Ordered sequence of price values (oldest first).
    title:
        Optional line printed above the chart (e.g. "TSLA – Last 30 Trading Days").
    height:
        Number of interior grid rows (excluding the x-axis).  Clamped to [3, 40].

    Returns
    -------
    A ready-to-print multi-line string.  Returns an error message string when
    fewer than 2 valid data points are present.
    """
    clean = [float(p) for p in prices if p is not None]
    if len(clean) < 2:
        return _error_message(title, len(clean))

    height = max(3, min(40, int(height)))
    low = min(clean)
    high = max(clean)
    price_range = high - low

    # Build a normalised row index for each price value (0 = bottom row, height-1 = top row).
    def _row(price: float) -> int:
        if price_range == 0:
            # Flat data: place everything in the middle row.
            return height // 2
        return round((price - low) / price_range * (height - 1))

    rows = [_row(p) for p in clean]
    n = len(clean)

    # --- y-axis label width ---
    label_high = f"{high:.2f}"
    label_low = f"{low:.2f}"
    y_label_width = max(len(label_high), len(label_low)) + 1  # + space after label

    # --- build empty grid ---
    # grid[row_idx][col_idx]  row_idx=0 is the BOTTOM row
    grid: List[List[str]] = [[" "] * n for _ in range(height)]

    for col, row_idx in enumerate(rows):
        grid[row_idx][col] = _POINT_CHAR

    # --- assemble output lines (top to bottom) ---
    lines: List[str] = []

    if title:
        lines.append(title)

    stats = _build_stats_from_prices(clean)
    lines.append(_format_stats_line(stats))
    lines.append("")

    # Render grid rows top-to-bottom; y-labels on the side at top/mid/bottom.
    interesting_rows = {0, height - 1, height // 2}
    for row_idx in reversed(range(height)):
        if price_range:
            price_at_row_val = low + row_idx / (height - 1) * price_range
        else:
            price_at_row_val = low

        if row_idx in interesting_rows:
            y_label = f"{price_at_row_val:>{y_label_width - 1}.2f} "
        else:
            y_label = " " * y_label_width

        row_str = "".join(grid[row_idx])
        lines.append(f"{y_label}{_AXIS_V}{row_str}")

    # x-axis
    x_axis = " " * y_label_width + _ORIGIN + _AXIS_H * n
    lines.append(x_axis)

    # x-axis tick labels
    tick_positions = _x_ticks(n)
    tick_row = [" "] * n
    for pos, label in tick_positions.items():
        for i, ch in enumerate(label):
            if pos + i < n:
                tick_row[pos + i] = ch
    lines.append(" " * (y_label_width + 1) + "".join(tick_row))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_stats_from_prices(prices: Sequence[float]) -> dict:
    """Return a dict with latest_price, high, low, return_pct (or None)."""
    clean = [float(p) for p in prices if p is not None]
    if not clean:
        return {"latest_price": None, "high": None, "low": None, "return_pct": None}
    latest = clean[-1]
    high = max(clean)
    low = min(clean)
    if len(clean) < 2 or clean[0] == 0:
        return_pct = None
    else:
        return_pct = (latest - clean[0]) / abs(clean[0]) * 100.0
    return {"latest_price": latest, "high": high, "low": low, "return_pct": return_pct}


def _format_stats_line(stats: dict) -> str:
    latest = stats.get("latest_price")
    high = stats.get("high")
    low = stats.get("low")
    return_pct = stats.get("return_pct")
    sign = "+" if (return_pct or 0) >= 0 else ""
    return_str = f"{sign}{return_pct:.2f}%" if return_pct is not None else "N/A"
    return (
        f"  Latest: {latest:.2f}   "
        f"High: {high:.2f}   "
        f"Low: {low:.2f}   "
        f"Return: {return_str}"
    )


def _x_ticks(n: int, max_ticks: int = 6) -> dict:
    """Return {column_index: label_string} for evenly spaced x-axis ticks."""
    if n <= 0:
        return {}
    step = max(1, n // max_ticks)
    ticks: dict = {}
    for i in range(0, n, step):
        label = str(i + 1)  # 1-based day number
        ticks[i] = label
    return ticks


def _error_message(title: str, n_points: int) -> str:
    header = title or "Price Chart"
    if n_points == 0:
        return f"{header}\n  [No price data available]"
    return f"{header}\n  [Insufficient data: only {n_points} point(s) — need at least 2]"


def print_price_chart(
    prices: Sequence[float],
    title: str = "",
    height: int = _CHART_HEIGHT,
) -> None:
    """Convenience wrapper that prints the chart directly to stdout."""
    print(render_price_chart(prices, title=title, height=height))
