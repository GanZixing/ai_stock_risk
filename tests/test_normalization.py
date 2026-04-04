import unittest

from normalization import normalize_value
from schemas import FactorConfig


class TestNormalization(unittest.TestCase):
    def test_normalize_higher_worse(self):
        cfg = FactorConfig(weight=1, min_value=0.0, max_value=1.0, direction="higher_worse")
        self.assertEqual(normalize_value(0.25, cfg), 25.0)

    def test_normalize_lower_worse(self):
        cfg = FactorConfig(weight=1, min_value=0.0, max_value=1.0, direction="lower_worse")
        self.assertEqual(normalize_value(0.25, cfg), 75.0)

    def test_normalize_none_value(self):
        cfg = FactorConfig(weight=1, min_value=0.0, max_value=1.0)
        self.assertEqual(normalize_value(None, cfg), 0.0)


if __name__ == "__main__":
    unittest.main()
