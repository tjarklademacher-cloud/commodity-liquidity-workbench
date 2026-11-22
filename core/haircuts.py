"""Dynamic volatility- and hedge-sensitive inventory haircut engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from core.schemas import (
    COMMODITY_SPECS,
    CommoditySpec,
    InventoryPosition,
    InventoryValuation,
    MarketQuote,
)


def compute_inventory_valuation(
    position: InventoryPosition,
    market_quote: MarketQuote,
    spec: CommoditySpec,
    shock_pct: float = 0.0,
) -> InventoryValuation:
    """Value one inventory line and decompose its clamped haircut into 100%-summing components."""
    shocked_price = max(0.0, market_quote.spot_price * (1.0 + shock_pct))
    hedge_ratio = (
        float(np.clip(position.hedged_qty / position.physical_qty, 0.0, 1.0))
        if position.physical_qty > 0.0
        else 0.0
    )
    vol_addon = spec.vol_slope * max(0.0, market_quote.realized_vol_30d - spec.ref_vol)
    unhedged_penalty = (1.0 - hedge_ratio) * spec.unhedged_penalty_rate
    raw_haircut = spec.base_haircut + vol_addon + unhedged_penalty
    total_haircut = float(np.clip(raw_haircut, spec.min_haircut, spec.max_haircut))
    advance_rate = 1.0 - total_haircut

    gross_market_value = position.physical_qty * shocked_price
    eligible_collateral = gross_market_value * advance_rate
    haircut_dollars = gross_market_value - eligible_collateral
    hedged_notional = position.hedged_qty * shocked_price
    futures_contracts = position.hedged_qty / spec.contract_multiplier

    eff_base_haircut = min(spec.base_haircut, total_haircut)
    rem1 = max(0.0, total_haircut - eff_base_haircut)
    eff_vol_addon = min(vol_addon, rem1)
    rem2 = max(0.0, rem1 - eff_vol_addon)
    eff_unhedged_penalty = min(unhedged_penalty, rem2)
    eff_floor_adjustment = max(
        0.0, total_haircut - (eff_base_haircut + eff_vol_addon + eff_unhedged_penalty)
    )

    return InventoryValuation(
        ticker=position.ticker,
        name=spec.name,
        exchange=spec.exchange,
        unit=spec.unit,
        physical_qty=position.physical_qty,
        hedged_qty=position.hedged_qty,
        spot_price=market_quote.spot_price,
        shock_pct=shock_pct,
        shocked_price=shocked_price,
        realized_vol_30d=market_quote.realized_vol_30d,
        hedge_ratio=hedge_ratio,
        futures_contracts=futures_contracts,
        base_haircut=spec.base_haircut,
        vol_addon=vol_addon,
        unhedged_penalty=unhedged_penalty,
        raw_haircut=raw_haircut,
        total_haircut=total_haircut,
        advance_rate=advance_rate,
        was_clamped_min=raw_haircut < spec.min_haircut,
        was_clamped_max=raw_haircut > spec.max_haircut,
        eff_base_haircut=eff_base_haircut,
        eff_vol_addon=eff_vol_addon,
        eff_unhedged_penalty=eff_unhedged_penalty,
        eff_floor_adjustment=eff_floor_adjustment,
        gross_market_value=gross_market_value,
        haircut_dollars=haircut_dollars,
        eligible_collateral=eligible_collateral,
        hedged_notional=hedged_notional,
    )


def compute_portfolio_inventory(
    positions: Sequence[InventoryPosition],
    market_snapshot: Mapping[str, MarketQuote],
    specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS,
    shocks_map: Mapping[str, float] | None = None,
) -> list[InventoryValuation]:
    """Value every inventory position, applying optional per-ticker price shocks."""
    shocks = shocks_map or {}
    return [
        compute_inventory_valuation(
            pos, market_snapshot[pos.ticker], specs[pos.ticker], float(shocks.get(pos.ticker, 0.0))
        )
        for pos in positions
    ]


def inventory_valuations_to_dataframe(items: Sequence[InventoryValuation]) -> pd.DataFrame:
    """Flatten inventory valuations into a DataFrame (one row per commodity)."""
    return pd.DataFrame([item.model_dump() for item in items])
