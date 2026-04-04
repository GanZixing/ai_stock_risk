import unittest

from main import run_stock_risk_agent
from schemas import RESERVED_FACTORS


class TestMainCliPipeline(unittest.TestCase):
    def test_run_stock_risk_agent_contains_reserved_factors(self):
        output = run_stock_risk_agent(
            ticker="TSLA",
            benchmark="SPY",
            lookback=252,
            disable_llm=True,
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


if __name__ == "__main__":
    unittest.main()
