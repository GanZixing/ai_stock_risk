# US Stock Risk Agent

A modular, config-driven risk assessment system for US stocks with deterministic scoring and LLM-powered summaries.

## Project Structure

```
ai_store/
├── cli/                 # Command-line interface
├── config/              # Configuration loaders
├── data/                # Data adapters
├── llm/                 # LLM summary layer
├── models/              # Risk models
├── normalization/       # Score normalization utilities
├── schemas/             # Base classes and dataclasses
├── scoring/             # Score aggregation utilities
├── tests/               # Unit tests
├── sample_run.py        # Example usage script
└── README.md
```

## Installation

```bash
cd ai_store
pip install -e .
```

## Example Run Command

```bash
python sample_run.py
```

This runs both models on sample data and generates an LLM summary.

## Main CLI Entrypoint

```bash
python main.py --ticker TSLA --benchmark SPY --lookback 252
```

Supported options:

- `--ticker`
- `--benchmark`
- `--lookback`
- `--output-json`
- `--disable-llm`
- `--config`
- `--debug`
- `--no-real-market-data`

Example with JSON export:

```bash
python main.py --ticker TSLA --benchmark SPY --lookback 252 --output-json main_output_tsla.json
```

## Streamlit UI

Run the UI:

```bash
pip install streamlit
streamlit run ui/app.py
```

The UI includes a reusable price chart module that:

- fetches the last 30 trading days for the selected ticker using the existing market adapter
- prefers adjusted close and falls back to close
- shows latest price, 30-day high/low, and 30-day return
- displays a clear message when market data cannot be fetched

## JSON Output Schema

Top-level output fields:

- `ticker`
- `benchmark`
- `timestamp`
- `statistical_risk`
- `structural_risk`
- `llm_summary`
- `reserved_factors_status`

Per-model fields:

- `model_name`
- `score`
- `level`
- `factor_scores`
- `raw_metrics`
- `weights`
- `notes`

Reserved factor fields (`AAA`, `BBB`, `CCC`):

- `enabled`
- `weight`
- `raw_value`
- `normalized_score`
- `affects_final_score`

Concrete example output is available in `sample_output_aapl.json`.

## Sample JSON Output

```json
{
  "ticker": "AAPL",
  "statistical_risk": {
    "model_name": "StatisticalRiskModel",
    "overall_score": 19.37,
    "risk_level": "Low",
    "factor_breakdown": [
      {
        "factor": "annualized_volatility",
        "raw_value": 0.025,
        "normalized_score": 0.0,
        "weight": 15,
        "thresholds": {
          "min_value": 0.1,
          "max_value": 0.5,
          "direction": "higher_worse"
        }
      }
    ],
    "raw_metrics": {
      "annualized_volatility": 0.025,
      "beta": 0.8
    },
    "weights_used": {
      "annualized_volatility": 15
    }
  },
  "structural_risk": {
    "model_name": "StructuralRiskModel",
    "overall_score": 29.64,
    "risk_level": "Moderate",
    "factor_breakdown": [...],
    "raw_metrics": {...},
    "weights_used": {...}
  },
  "llm_summary": {
    "short_summary": "AAPL has a statistical risk score of 19.37 and structural risk score of 29.64.",
    "long_summary": "For AAPL, the Statistical Risk Model indicates a risk level of Low... The Structural Risk Model shows a risk level of Moderate...",
    "key_risk_drivers": ["Annualized Volatility", "Profitability Risk"],
    "model_disagreement_note": "Models disagree significantly.",
    "reserved_factor_note": "AAA, BBB, and CCC are reserved placeholder factors with zero impact on the scores.",
    "final_combined_interpretation": "Overall, AAPL exhibits low combined risk."
  }
}
```

## CLI Usage

```bash
python -m cli AAPL --price-data price.json --financial-data financial.json --company-profile profile.json --llm-summary
```

## Reserved Factors

AAA, BBB, CCC are placeholder factors with zero weight, included for future extensibility.

- They are globally registered in the factor registry and always included in input/output schemas.
- They are always emitted in factor breakdown and reserved factor status output for transparency.
- Current default behavior is inactive: weight = 0 and normalized score = 0.
- They do not impact final scores unless explicitly activated.
- Future activation is designed to be config-only (weight/threshold updates plus optional value source), with no model refactor required.

## Architecture

- **Modular**: Separate concerns for data, models, scoring, etc.
- **Config-driven**: Thresholds and weights loaded from config.
- **Deterministic**: Core scoring uses formulas, no randomness.
- **Agent-friendly**: JSON outputs for easy integration.
- **Extensible**: Base interfaces for adding new factors/models.


## Custom configuration

You can override factor thresholds and weights by creating a custom `StatisticalRiskModelConfig`.

## Notes

- `AAA`, `BBB`, and `CCC` are configured as placeholder factors with default weight `0`.
- The model is designed for deterministic calculations only.
- The adapter interface `RiskDataAdapter` supports future integration of external price providers.
