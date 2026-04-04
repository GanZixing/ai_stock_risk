from statistical_risk_model import StatisticalRiskModel


def main() -> None:
    asset_history = [
        {"adj_close": price, "volume": 1000000 + idx * 5000}
        for idx, price in enumerate([150.0, 151.5, 149.0, 152.0, 153.0, 151.0])
    ]
    benchmark_history = [
        {"adj_close": price, "volume": 3000000 + idx * 10000}
        for idx, price in enumerate([420.0, 421.0, 419.0, 423.0, 425.0, 426.0])
    ]

    model = StatisticalRiskModel()
    result = model.evaluate(
        ticker="TEST",
        benchmark_ticker="SPY",
        lookback_window=6,
        price_history=asset_history,
        benchmark_history=benchmark_history,
    )

    print("Model Name:", result.model_name)
    print("Overall Score:", result.overall_score)
    print("Risk Level:", result.risk_level)
    print("Weights Used:", result.weights_used)
    print("Raw Metrics:")
    for key, value in result.raw_metrics.items():
        print(f"  {key}: {value}")
    print("Factor Breakdown:")
    for factor in result.factor_breakdown:
        print(
            f"  {factor.factor}: raw={factor.raw_value}, normalized={factor.normalized_score}, weight={factor.weight}"
        )
    print("Explanation:", result.explanation)


if __name__ == "__main__":
    main()
