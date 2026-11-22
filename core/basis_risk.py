"""Physical-versus-futures pricing and illustrative basis scenarios.

Definition used throughout: basis = physical price - futures price, expressed as a share of the futures
price. Physical price = futures price x (1 + basis). Futures prices drive variation margin; physical prices
(after any liquidation discount) drive inventory collateral and the borrowing base.
"""

from __future__ import annotations

from collections.abc import Mapping

from core.advanced_schemas import BasisAssumption
from core.schemas import REQUIRED_TICKERS

BASIS_PRESETS: dict[str, dict[str, dict[str, float]]] = {
    "A. Stable basis": {"basis_shocks": {}, "liquidation_discounts": {}},
    "B. Weakening basis (physical lags futures by 5%)": {
        "basis_shocks": {t: -0.05 for t in REQUIRED_TICKERS}, "liquidation_discounts": {}},
    "C. Strengthening basis (physical outpaces futures by 5%)": {
        "basis_shocks": {t: 0.05 for t in REQUIRED_TICKERS}, "liquidation_discounts": {}},
    "D. Adverse liquidation (8% discount to futures)": {
        "basis_shocks": {}, "liquidation_discounts": {t: 0.08 for t in REQUIRED_TICKERS}},
    "E. Cocoa dislocation (cocoa basis -15%)": {"basis_shocks": {"CC=F": -0.15}, "liquidation_discounts": {}},
}
"""Illustrative basis scenarios; the magnitudes are assumptions, not observed market basis."""


def default_basis_assumptions() -> dict[str, BasisAssumption]:
    """Neutral opening basis (physical priced at the futures reference)."""
    return {t: BasisAssumption(ticker=t, opening_basis_pct=0.0) for t in REQUIRED_TICKERS}


def physical_price(futures_price: float, basis_pct: float) -> float:
    return max(0.0, futures_price * (1.0 + basis_pct))


def scenario_prices(
    futures_price_0: float,
    futures_shock: float,
    opening_basis_pct: float,
    basis_shock: float,
    liquidation_discount: float,
) -> tuple[float, float, float, float, float]:
    """(F0, F1, P0, P1, collateral price) for one commodity, all USD per physical unit and non-negative."""
    f0 = futures_price_0
    f1 = max(0.0, f0 * (1.0 + futures_shock))
    p0 = physical_price(f0, opening_basis_pct)
    p1 = physical_price(f1, opening_basis_pct + basis_shock)
    collateral = p1 * (1.0 - liquidation_discount)
    return f0, f1, p0, p1, collateral


def economic_pnl(
    physical_qty: float,
    signed_contracts: float,
    contract_multiplier: float,
    f0: float,
    f1: float,
    p0: float,
    p1: float,
    opening_basis_pct: float,
    liquidation_discount: float,
) -> dict[str, float]:
    """Economic P&L split. physical = residual price + basis + futures offset; combined excludes cash timing.

    physical_pnl      = Q x (P1 - P0)
    futures_pnl       = signed contracts x size x (F1 - F0)        (equals variation margin cash)
    basis_pnl         = Q x F1 x (basis_1 - basis_0)
    residual_price    = Q x (F1 - F0) x (1 + basis_0) + futures_pnl
    combined          = physical_pnl + futures_pnl = residual_price + basis_pnl
    liquidation cost  = Q x P1 x discount  (reported separately; reduces collateral, not the mark)
    """
    physical = physical_qty * (p1 - p0)
    futures = signed_contracts * contract_multiplier * (f1 - f0)
    basis_1 = (p1 / f1 - 1.0) if f1 > 0 else opening_basis_pct
    basis = physical_qty * f1 * (basis_1 - opening_basis_pct)
    residual = physical_qty * (f1 - f0) * (1.0 + opening_basis_pct) + futures
    return {
        "physical_pnl": physical,
        "futures_pnl": futures,
        "basis_pnl": basis,
        "residual_price_pnl": residual,
        "combined_pnl": physical + futures,
        "liquidation_discount_cost": physical_qty * p1 * liquidation_discount,
    }


def opening_basis_map(basis: Mapping[str, BasisAssumption]) -> dict[str, float]:
    return {t: basis[t].opening_basis_pct for t in REQUIRED_TICKERS}
