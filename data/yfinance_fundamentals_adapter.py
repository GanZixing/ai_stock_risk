from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

from data import FinancialDataAdapter


class YFinanceFundamentalsAdapter(FinancialDataAdapter):
    """Fetches company fundamentals/profile data from yfinance."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def get_company_data(self, ticker: str) -> Mapping[str, Any]:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("yfinance is required for YFinanceFundamentalsAdapter") from exc

        company = yf.Ticker(ticker)
        info = self._as_mapping(getattr(company, "info", {}) or {})

        financials = self._safe_frame(company, "financials")
        balance_sheet = self._safe_frame(company, "balance_sheet")
        cashflow = self._safe_frame(company, "cashflow")

        income_statement = {
            "net_income": self._first_number(
                self._series_values(financials, ["Net Income", "NetIncome"]) + [info.get("netIncomeToCommon")]
            ),
            "revenue": self._first_number(
                self._series_values(financials, ["Total Revenue", "TotalRevenue"]) + [info.get("totalRevenue")]
            ),
            "gross_margin": self._as_float(info.get("grossMargins")),
            "operating_margin": self._as_float(info.get("operatingMargins")),
            "roe": self._as_float(info.get("returnOnEquity")),
            "roa": self._as_float(info.get("returnOnAssets")),
            "revenue_history": self._history_values(financials, ["Total Revenue", "TotalRevenue"]),
            "eps_history": self._eps_history(info),
            "net_income_history": self._history_values(financials, ["Net Income", "NetIncome"]),
        }

        balance_data = {
            "total_debt": self._first_number(
                self._series_values(balance_sheet, ["Total Debt", "TotalDebt"]) + [info.get("totalDebt")]
            ),
            "total_equity": self._first_number(
                self._series_values(
                    balance_sheet,
                    [
                        "Stockholders Equity",
                        "StockholdersEquity",
                        "Total Stockholder Equity",
                        "TotalStockholderEquity",
                        "Total Equity Gross Minority Interest",
                    ],
                )
                + [info.get("totalStockholderEquity")]
            ),
            "current_assets": self._first_number(self._series_values(balance_sheet, ["Current Assets", "CurrentAssets"])),
            "current_liabilities": self._first_number(
                self._series_values(balance_sheet, ["Current Liabilities", "CurrentLiabilities"])
            ),
            "interest_expense": self._first_number(
                self._series_values(financials, ["Interest Expense", "InterestExpense"])
            ),
            "ebit": self._first_number(self._series_values(financials, ["EBIT", "Ebit"])),
        }

        cash_flow = {
            "operating_cash_flow": self._first_number(
                self._series_values(cashflow, ["Operating Cash Flow", "OperatingCashFlow"])
            ),
            "free_cash_flow": self._first_number(
                self._series_values(cashflow, ["Free Cash Flow", "FreeCashFlow"]) + [info.get("freeCashflow")]
            ),
            "cash_flow_from_ops_history": self._history_values(cashflow, ["Operating Cash Flow", "OperatingCashFlow"]),
            "free_cash_flow_history": self._history_values(cashflow, ["Free Cash Flow", "FreeCashFlow"]),
        }

        market = {
            "market_cap": self._as_float(info.get("marketCap")),
            "pe_ratio": self._as_float(info.get("trailingPE")),
            "ps_ratio": self._as_float(info.get("priceToSalesTrailing12Months")),
            "peg_ratio": self._as_float(info.get("pegRatio")),
        }

        company_profile = {
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "customer_concentration": None,
            "product_concentration": None,
        }

        sector = company_profile.get("sector")
        industry_metadata = {
            "sector_risk_multiplier": {sector: 1.0} if sector else {},
        }

        return {
            "financial_data": {
                "income_statement": income_statement,
                "balance_sheet": balance_data,
                "cash_flow": cash_flow,
                "market": market,
            },
            "company_profile": company_profile,
            "industry_metadata": industry_metadata,
            "source_metadata": {
                "provider": "yfinance",
                "ticker": ticker,
            },
        }

    @staticmethod
    def _as_mapping(value: Any) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        return {}

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            parsed = float(value)
            if math.isnan(parsed):
                return None
            return parsed
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_frame(company: Any, attr: str) -> Any:
        try:
            return getattr(company, attr)
        except Exception:
            return None

    def _series_values(self, frame: Any, candidates: Sequence[str]) -> List[Optional[float]]:
        if frame is None or getattr(frame, "empty", True):
            return []
        for row_name in candidates:
            try:
                if row_name in frame.index:
                    row = frame.loc[row_name]
                    if hasattr(row, "sort_index"):
                        row = row.sort_index()
                    values: List[Optional[float]] = []
                    for value in row.tolist():
                        values.append(self._as_float(value))
                    return values
            except Exception:
                continue
        return []

    def _history_values(self, frame: Any, candidates: Sequence[str], max_points: int = 3) -> List[float]:
        values = [v for v in self._series_values(frame, candidates) if v is not None]
        if not values:
            return []
        return values[-max_points:]

    def _first_number(self, values: Sequence[Any]) -> Optional[float]:
        for value in values:
            as_float = self._as_float(value)
            if as_float is not None:
                return as_float
        return None

    def _eps_history(self, info: Mapping[str, Any]) -> List[float]:
        # yfinance does not consistently provide historical EPS series in .info.
        trailing = self._as_float(info.get("trailingEps"))
        forward = self._as_float(info.get("forwardEps"))
        values = [v for v in [trailing, forward] if v is not None]
        return values