from typing import Optional

from schemas import FactorConfig


def normalize_value(raw: Optional[float], config: FactorConfig) -> float:
    if raw is None:
        return 0.0
    min_val = config.min_value
    max_val = config.max_value
    if max_val == min_val:
        return 0.0
    ratio = (raw - min_val) / (max_val - min_val)
    if config.direction == "lower_worse":
        ratio = 1.0 - ratio
    score = max(0.0, min(1.0, ratio)) * 100.0
    return round(score, 2)
