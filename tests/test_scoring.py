import unittest

from scoring import aggregate_score, risk_level_label
from schemas import FactorResult


class TestScoring(unittest.TestCase):
    def test_weighted_aggregation(self):
        factors = [
            FactorResult("f1", raw_value=1.0, normalized_score=20.0, weight=10.0, thresholds={}),
            FactorResult("f2", raw_value=1.0, normalized_score=60.0, weight=30.0, thresholds={}),
        ]
        # (20*10 + 60*30) / (10+30) = 50
        self.assertEqual(aggregate_score(factors), 50.0)

    def test_aggregation_skips_none_raw_values(self):
        factors = [
            FactorResult("f1", raw_value=None, normalized_score=100.0, weight=50.0, thresholds={}),
            FactorResult("f2", raw_value=1.0, normalized_score=40.0, weight=50.0, thresholds={}),
        ]
        self.assertEqual(aggregate_score(factors), 40.0)

    def test_risk_level_mapping(self):
        self.assertEqual(risk_level_label(25), "Low")
        self.assertEqual(risk_level_label(26), "Moderate")
        self.assertEqual(risk_level_label(51), "High")
        self.assertEqual(risk_level_label(76), "Extreme")


if __name__ == "__main__":
    unittest.main()
