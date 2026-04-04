from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Sequence


class PriceDataAdapter(ABC):
    """Adapter interface for historical price data."""

    @abstractmethod
    def get_price_history(
        self,
        ticker: str,
        lookback_window: int,
        benchmark_ticker: Optional[str] = None,
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Return a mapping with at least 'asset' and 'benchmark' time series."""
        raise NotImplementedError


class FinancialDataAdapter(ABC):
    """Adapter interface for company-level structural risk data."""

    @abstractmethod
    def get_company_data(
        self,
        ticker: str,
    ) -> Mapping[str, Any]:
        """Return a mapping with financials, profile, and optional metadata."""
        raise NotImplementedError
