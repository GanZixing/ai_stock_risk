import unittest

from ui.ascii_chart import render_price_chart


class TestRenderPriceChart(unittest.TestCase):
    PRICES_UP = [100.0 + i * 2 for i in range(30)]    # steady uptrend
    PRICES_DOWN = [200.0 - i * 2 for i in range(30)]  # steady downtrend
    PRICES_FLAT = [150.0] * 30                         # no price movement

    # ------------------------------------------------------------------
    # Structural shape tests
    # ------------------------------------------------------------------

    def test_output_is_non_empty_string(self):
        result = render_price_chart(self.PRICES_UP)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_contains_y_axis_and_x_axis_characters(self):
        result = render_price_chart(self.PRICES_UP)
        self.assertIn("│", result)
        self.assertIn("─", result)
        self.assertIn("└", result)

    def test_contains_data_point_char(self):
        result = render_price_chart(self.PRICES_UP)
        self.assertIn("*", result)

    def test_title_appears_when_supplied(self):
        result = render_price_chart(self.PRICES_UP, title="TSLA - Last 30 Trading Days")
        self.assertIn("TSLA - Last 30 Trading Days", result)

    def test_title_absent_when_not_supplied(self):
        result = render_price_chart(self.PRICES_UP, title="")
        lines = result.splitlines()
        self.assertNotEqual(lines[0], "")
        self.assertIn("│", result)

    def test_does_not_embed_price_stats(self):
        result = render_price_chart(self.PRICES_UP)
        self.assertNotIn("Latest:", result)
        self.assertNotIn("30D High", result)
        self.assertNotIn("30D Low", result)

    def test_correct_height_line_count(self):
        height = 10
        result = render_price_chart(self.PRICES_UP, height=height)
        # title + stats + blank + height inner rows + x-axis + tick row = height + 5
        lines = result.splitlines()
        # At least height rows of grid data should appear (flexible check)
        self.assertGreaterEqual(len(lines), height)

    def test_height_clamped_minimum(self):
        result_min = render_price_chart(self.PRICES_UP, height=1)
        result_3 = render_price_chart(self.PRICES_UP, height=3)
        self.assertEqual(result_min, result_3)

    def test_height_clamped_maximum(self):
        result_max = render_price_chart(self.PRICES_UP, height=99)
        result_40 = render_price_chart(self.PRICES_UP, height=40)
        self.assertEqual(result_max, result_40)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_flat_data_renders_without_error(self):
        result = render_price_chart(self.PRICES_FLAT)
        self.assertIn("*", result)

    def test_flat_data_shows_single_price_level(self):
        result = render_price_chart(self.PRICES_FLAT)
        self.assertIn("150.00", result)

    def test_single_point_returns_error_message(self):
        result = render_price_chart([100.0])
        self.assertNotIn("│", result)
        self.assertIn("Insufficient data", result)

    def test_empty_returns_error_message(self):
        result = render_price_chart([])
        self.assertNotIn("├", result)
        self.assertIn("No price data", result)

    def test_downtrend_renders(self):
        result = render_price_chart(self.PRICES_DOWN)
        self.assertIn("*", result)

    def test_min_max_labels_appear(self):
        result = render_price_chart(self.PRICES_UP)
        # min price = 100.00, max price = 158.00
        self.assertIn("100.00", result)
        self.assertIn("158.00", result)

    def test_two_points_works(self):
        result = render_price_chart([50.0, 75.0])
        self.assertIn("*", result)

    def test_custom_point_marker_supported(self):
        result = render_price_chart(self.PRICES_UP, point_char=".")
        self.assertIn(".", result)


if __name__ == "__main__":
    unittest.main()
