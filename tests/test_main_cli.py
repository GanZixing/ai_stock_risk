import unittest

from main import run_stock_risk_agent
from schemas import RESERVED_FACTORS


class TestMainCliPipeline(unittest.TestCase):
    def test_run_stock_risk_agent_contains_reserved_factors(self):
        output, _debug = run_stock_risk_agent(
            ticker="TSLA",
            benchmark="SPY",
            lookback=252,
            disable_llm=True,
            prefer_real_market_data=False,
        )

        self.assertEqual(output["ticker"], "TSLA")
        self.assertEqual(output["benchmark"], "SPY")
        self.assertIn("statistical_risk", output)
        self.assertIn("structural_risk", output)
        self.assertIn("reserved_factors_status", output)

        for factor in RESERVED_FACTORS:
            self.assertIn(factor, output["reserved_factors_status"])
            factor_status = output["reserved_factors_status"][factor]
            self.assertFalse(factor_status["enabled"])
            self.assertEqual(factor_status["weight"], 0.0)
            self.assertEqual(factor_status["normalized_score"], 0.0)
            self.assertFalse(factor_status["affects_final_score"])

    def test_run_stock_risk_agent_includes_market_fetch_debug_logs(self):
        _output, debug = run_stock_risk_agent(
            ticker="TSLA",
            benchmark="SPY",
            lookback=252,
            disable_llm=True,
            debug=False,
            prefer_real_market_data=False,
        )

        self.assertIn("data_source", debug)
        self.assertIn("market_fetch_debug", debug["data_source"])

        market_debug = debug["data_source"]["market_fetch_debug"]
        self.assertIn("asset", market_debug)
        self.assertIn("benchmark", market_debug)
        self.assertIn("rows", market_debug["asset"])
        self.assertIn("date_start", market_debug["asset"])
        self.assertIn("date_end", market_debug["asset"])
        self.assertIn("first_price", market_debug["asset"])
        self.assertIn("last_price", market_debug["asset"])

    def test_run_stock_risk_agent_includes_structural_source_and_inputs(self):
        _output, debug = run_stock_risk_agent(
            ticker="TSLA",
            benchmark="SPY",
            lookback=252,
            disable_llm=True,
            debug=False,
            prefer_real_market_data=False,
        )

        self.assertIn("data_source", debug)
        self.assertIn("structural_data_source", debug["data_source"])
        self.assertEqual(debug["data_source"]["structural_data_source"], "local_sample_fallback")

        self.assertIn("structural", debug)
        self.assertIn("input_fields", debug["structural"])
        input_fields = debug["structural"]["input_fields"]
        self.assertIn("income_statement", input_fields)
        self.assertIn("balance_sheet", input_fields)
        self.assertIn("cash_flow", input_fields)
        self.assertIn("market", input_fields)
        self.assertIn("company_profile", input_fields)


if __name__ == "__main__":
    unittest.main()
