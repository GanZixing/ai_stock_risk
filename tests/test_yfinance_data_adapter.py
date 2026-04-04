import unittest
from datetime import datetime

from statistical_risk_model import YFinanceDataAdapter


class FakeFrame:
    def __init__(self, rows):
        self._rows = rows
        self.empty = len(rows) == 0

    def iterrows(self):
        for idx, row in self._rows:
            yield idx, row


class TestYFinanceDataAdapter(unittest.TestCase):
    def test_aligns_dates_and_drops_missing(self):
        adapter = YFinanceDataAdapter()

        asset_rows = [
            (datetime(2024, 1, 1), {"Adj Close": 100.0, "Close": 100.0, "Volume": 1000}),
            (datetime(2024, 1, 2), {"Adj Close": None, "Close": 101.0, "Volume": 1100}),
            (datetime(2024, 1, 3), {"Adj Close": 102.0, "Close": 102.0, "Volume": 1200}),
            (datetime(2024, 1, 4), {"Adj Close": 103.0, "Close": None, "Volume": 1300}),
        ]
        benchmark_rows = [
            (datetime(2024, 1, 1), {"Adj Close": 200.0, "Close": 200.0, "Volume": 2000}),
            (datetime(2024, 1, 2), {"Adj Close": 201.0, "Close": 201.0, "Volume": 2100}),
            (datetime(2024, 1, 3), {"Adj Close": 202.0, "Close": 202.0, "Volume": 2200}),
            (datetime(2024, 1, 5), {"Adj Close": 204.0, "Close": 204.0, "Volume": 2400}),
        ]

        def fake_download(ticker, lookback_window):
            if ticker == "TSLA":
                return FakeFrame(asset_rows)
            return FakeFrame(benchmark_rows)

        adapter._download_history = fake_download

        result = adapter.get_price_history("TSLA", 252, "SPY")

        self.assertIn("asset", result)
        self.assertIn("benchmark", result)
        self.assertEqual(len(result["asset"]), len(result["benchmark"]))
        self.assertEqual(len(result["asset"]), 3)

        for row in result["asset"]:
            self.assertIn("date", row)
            self.assertIn("adj_close", row)
            self.assertIn("close", row)
            self.assertIn("volume", row)

    def test_uses_fallback_on_failure(self):
        fallback = {
            "asset": [{"date": "2024-01-01", "adj_close": 1.0, "close": 1.0, "volume": 1.0}],
            "benchmark": [{"date": "2024-01-01", "adj_close": 1.0, "close": 1.0, "volume": 1.0}],
        }
        adapter = YFinanceDataAdapter(fallback_data=fallback)

        def fail_download(_ticker, _lookback):
            raise RuntimeError("API unavailable")

        adapter._download_history = fail_download
        result = adapter.get_price_history("TSLA", 252, "SPY")
        self.assertEqual(result, fallback)


if __name__ == "__main__":
    unittest.main()
