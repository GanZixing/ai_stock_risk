import unittest

from ui.chart_service import build_chart_summary_lines, get_chart_time_series


class StubAdapter:
    def __init__(self, payload=None, error=None):
        self.payload = payload or {}
        self.error = error

    def get_price_history(self, ticker, lookback_window, benchmark_ticker=None):
        if self.error is not None:
            raise self.error
        return self.payload


class TestChartService(unittest.TestCase):
    def test_get_chart_time_series_returns_points_and_stats(self):
        payload = {
            "asset": [
                {"date": "2026-03-28", "adj_close": 100.0},
                {"date": "2026-03-29", "adj_close": 110.0},
                {"date": "2026-03-30", "adj_close": 105.0},
            ]
        }
        result = get_chart_time_series(
            "TSLA",
            lookback_days=30,
            data_adapter=StubAdapter(payload=payload),
            intraday_fetcher=lambda _ticker: {"has_intraday": False, "reason": "none", "market_in_progress": False},
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["title"], "TSLA - Last 30 Trading Days")
        self.assertEqual(len(result["points"]), 3)
        self.assertEqual(result["stats"]["latest_price"], 105.0)
        self.assertEqual(result["stats"]["high_30d"], 110.0)
        self.assertEqual(result["stats"]["low_30d"], 100.0)
        self.assertAlmostEqual(result["stats"]["return_30d_pct"], 5.0)
        self.assertIn("summary_lines", result)

    def test_get_chart_time_series_uses_close_when_adj_close_missing(self):
        payload = {
            "asset": [
                {"date": "2026-03-28", "close": 100.0},
                {"date": "2026-03-29", "adj_close": None, "close": 98.0},
                {"date": "2026-03-30", "adj_close": 101.0},
                {"date": "2026-03-31", "adj_close": None, "close": None},
            ]
        }

        result = get_chart_time_series(
            "TSLA",
            lookback_days=30,
            data_adapter=StubAdapter(payload=payload),
            intraday_fetcher=lambda _ticker: {"has_intraday": False, "reason": "none", "market_in_progress": False},
        )
        self.assertTrue(result["success"])
        self.assertEqual([p["price"] for p in result["points"]], [100.0, 98.0, 101.0])

    def test_get_chart_time_series_returns_clear_error_when_fetch_fails(self):
        result = get_chart_time_series(
            "TSLA",
            lookback_days=30,
            data_adapter=StubAdapter(error=RuntimeError("API unavailable")),
            intraday_fetcher=lambda _ticker: {"has_intraday": False, "reason": "none", "market_in_progress": False},
        )

        self.assertFalse(result["success"])
        self.assertIn("Unable to fetch market data for TSLA", result["error"])
        self.assertEqual(result["points"], [])
        self.assertIsNone(result["stats"])

    def test_build_chart_summary_lines_with_intraday(self):
        stats = {
            "latest_price": 360.59,
            "high_30d": 417.40,
            "low_30d": 355.28,
            "return_30d_pct": -12.44,
        }
        intraday = {
            "has_intraday": True,
            "market_in_progress": True,
            "premarket_high": 362.10,
            "premarket_low": 358.80,
            "regular_high": 361.50,
            "regular_low": 359.20,
            "after_hours_high": None,
            "after_hours_low": None,
        }
        lines = build_chart_summary_lines(stats=stats, intraday=intraday)
        rendered = "\n".join(lines)
        self.assertIn("Latest: 360.59", rendered)
        self.assertIn("30D High: 417.40", rendered)
        self.assertIn("30D Low : 355.28", rendered)
        self.assertIn("Intraday Session:", rendered)
        self.assertIn("After-hours High: N/A", rendered)


if __name__ == "__main__":
    unittest.main()