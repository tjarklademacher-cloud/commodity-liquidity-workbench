"""Exchange-style futures margin: integer contract conversion, per-contract initial margin and VM.

This is a transparent approximation, not an exchange's clearing model (SPAN, IRM or similar). Default margin
per contract is an illustrative share of contract notional at the bundled sample price; users can replace it
with figures from the relevant exchange or clearing broker.

Conventions
- A hedge is a short futures position. Signed contract counts are negative for shorts.
- Variation margin cash to the merchant = signed contracts x contract size x change in futures price
  (USD per physical unit), so a short pays when futures rise.
- Initial margin required = |contracts| x IM per contract x stress multiplier + fixed add-on
  (the multiplier is applied first; the fixed add-on is not multiplied).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from core.advanced_schemas import ContractPolicy, MarginProfile, MarginSettings
from core.data_feed import SAMPLE_CALIBRATION
from core.schemas import COMMODITY_SPECS, REQUIRED_TICKERS, CommoditySpec

# Illustrative initial margin as a share of one contract's notional at the bundled sample price.
ILLUSTRATIVE_IM_SHARE: dict[str, float] = {
    "GC=F": 0.10, "HG=F": 0.08, "LBR=F": 0.12, "CC=F": 0.15, "KC=F": 0.10, "SB=F": 0.08,
}
ILLUSTRATIVE_MAINTENANCE_RATIO = 0.90
SAMPLE_REFERENCE_DATE = "2026-11-20"


def sample_reference_price(ticker: str, specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS) -> float:
    """Terminal bundled-sample price in USD per physical unit (cents-quoted contracts converted)."""
    return SAMPLE_CALIBRATION[ticker]["target_end"] * specs[ticker].quote_scale


def default_margin_settings(specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS) -> MarginSettings:
    profiles = {}
    for t in REQUIRED_TICKERS:
        share = ILLUSTRATIVE_IM_SHARE[t]
        notional = specs[t].contract_multiplier * sample_reference_price(t, specs)
        im = float(round(share * notional, -2))
        profiles[t] = MarginProfile(
            ticker=t, im_per_contract=im, maintenance_per_contract=float(round(im * ILLUSTRATIVE_MAINTENANCE_RATIO, -2)),
            source=f"Illustrative: {share:.0%} of contract notional at the bundled sample price",
            source_date=SAMPLE_REFERENCE_DATE, assumption_type="illustrative",
        )
    return MarginSettings(profiles=profiles)


def contracts_for_hedge(hedged_qty: float, contract_multiplier: float, policy: ContractPolicy) -> tuple[float, float]:
    """(exact contract count, traded contract count) for a hedge in physical units; both non-negative.

    `continuous` keeps fractional contracts (the legacy model's convention); the other policies round to
    whole exchange contracts, leaving a residual hedge quantity that is reported, not hidden.
    """
    exact = max(0.0, hedged_qty) / contract_multiplier
    if policy == "continuous":
        return exact, exact
    if policy == "floor":
        return exact, float(math.floor(exact + 1e-9))
    if policy == "ceil":
        return exact, float(math.ceil(exact - 1e-9))
    return exact, float(math.floor(exact + 0.5))


def signed_short_contracts(hedged_qty: float, ticker: str, settings: MarginSettings,
                           specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS) -> tuple[float, float, float]:
    """(signed contracts, exact contracts, residual hedge qty in physical units) for a short hedge."""
    mult = specs[ticker].contract_multiplier
    exact, traded = contracts_for_hedge(hedged_qty, mult, settings.contract_policy)
    residual = max(0.0, hedged_qty) - traded * mult
    return -traded, exact, residual


def variation_margin(signed_contracts: float, contract_multiplier: float, price_change: float) -> float:
    """Cash to the merchant over one settlement interval (negative = payment)."""
    return signed_contracts * contract_multiplier * price_change


def initial_margin_required(signed_contracts: float, profile: MarginProfile, multiplier: float = 1.0) -> float:
    if signed_contracts == 0.0:
        return 0.0
    return abs(signed_contracts) * profile.im_per_contract * max(0.0, multiplier) + profile.fixed_addon_usd


def im_change(posted: float, required: float, release_excess: bool) -> tuple[float, float]:
    """(additional IM to post, IM released back) when the requirement moves from posted to required."""
    additional = max(0.0, required - posted)
    released = max(0.0, posted - required) if release_excess else 0.0
    return additional, released
