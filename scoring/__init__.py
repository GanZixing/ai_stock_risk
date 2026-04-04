from typing import Iterable

from schemas import FactorResult


def aggregate_score(factors: Iterable[FactorResult]) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for factor in factors:
        if factor.raw_value is None:
            continue
        weighted_sum += factor.normalized_score * factor.weight
        total_weight += factor.weight
    if total_weight <= 0:
        return 0.0
    return weighted_sum / total_weight


def risk_level_label(score: float) -> str:
    if score <= 25:
        return "Low"
    if score <= 50:
        return "Moderate"
    if score <= 75:
        return "High"
    return "Extreme"
