"""Pydantic v2 schemas, enums, commodity specifications and calibrated defaults."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REQUIRED_TICKERS: tuple[str, ...] = ("GC=F", "HG=F", "LBR=F", "CC=F")
CAP_TIER_OPTIONS: tuple[float, ...] = (0.15, 0.175, 0.20)


class LiquidityRegime(StrEnum):
    HEALTHY = "HEALTHY"
    MARGIN_TRAP_WARNING = "MARGIN_TRAP_WARNING"
    BB_DEFICIENCY_WARNING = "BB_DEFICIENCY_WARNING"
    CRITICAL_SQUEEZE = "CRITICAL_SQUEEZE"


class CommoditySpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    name: str
    exchange: str
    unit: str
    contract_multiplier: float = Field(gt=0.0)
    base_haircut: float = Field(ge=0.0, le=1.0)
    ref_vol: float = Field(ge=0.0)
    vol_slope: float = Field(ge=0.0)
    unhedged_penalty_rate: float = Field(ge=0.0, le=1.0)
    min_haircut: float = Field(ge=0.0, le=1.0)
    max_haircut: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_haircut_bounds(self) -> CommoditySpec:
        if self.min_haircut > self.max_haircut:
            raise ValueError(
                f"min_haircut ({self.min_haircut}) must be <= max_haircut ({self.max_haircut})"
            )
        return self


class MarketQuote(BaseModel):
    ticker: str
    spot_price: float = Field(gt=0.0)
    prev_price: float = Field(gt=0.0)
    ret_1d: float
    ret_30d: float
    realized_vol_30d: float = Field(ge=0.0)
    vol_30d_min: float = Field(ge=0.0)
    vol_30d_max: float = Field(ge=0.0)
    as_of_date: str
    vol_fallback_used: bool = False


class MarketLoadResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prices_df: pd.DataFrame
    vol_df: pd.DataFrame
    snapshot: dict[str, MarketQuote]
    source_label: str
    as_of_date: str
    used_fallback: bool = False
    status_message: str = ""


class InventoryPosition(BaseModel):
    ticker: str
    physical_qty: float = Field(ge=0.0)
    hedged_qty: float = Field(ge=0.0)

    @field_validator("ticker")
    @classmethod
    def _check_ticker(cls, value: str) -> str:
        if value not in REQUIRED_TICKERS:
            raise ValueError(f"ticker must be one of {REQUIRED_TICKERS}, got {value!r}")
        return value


class ReceivableItem(BaseModel):
    buyer_name: str = Field(min_length=1)
    commodity: str = Field(min_length=1)
    gross_amount: float = Field(ge=0.0)
    days_past_due: int = Field(ge=0)
    is_investment_grade: bool


class FacilityTerms(BaseModel):
    facility_limit: float = Field(gt=0.0, default=115_000_000.0)
    drawn_revolver: float = Field(ge=0.0, default=109_000_000.0)
    unrestricted_cash: float = Field(ge=0.0, default=4_500_000.0)
    baseline_initial_margin: float = Field(ge=0.0, default=4_800_000.0)
    concentration_cap_pct: float = Field(gt=0.0, le=1.0, default=0.15)
    im_stepup_trigger: float = Field(ge=0.0, le=1.0, default=0.10)
    im_stepup_fraction: float = Field(ge=0.0, le=2.0, default=0.20)
    ar_aging_limit_days: int = Field(ge=0, default=90)
    ar_ig_advance_rate: float = Field(ge=0.0, le=1.0, default=0.85)
    ar_non_ig_advance_rate: float = Field(ge=0.0, le=1.0, default=0.75)
    default_active_shock_pct: float = Field(ge=-0.50, le=0.50, default=0.14)


class InventoryValuation(BaseModel):
    ticker: str
    name: str
    exchange: str
    unit: str
    physical_qty: float
    hedged_qty: float
    spot_price: float
    shock_pct: float
    shocked_price: float
    realized_vol_30d: float
    hedge_ratio: float
    futures_contracts: float
    base_haircut: float
    vol_addon: float
    unhedged_penalty: float
    raw_haircut: float
    total_haircut: float
    advance_rate: float
    was_clamped_min: bool
    was_clamped_max: bool
    eff_base_haircut: float
    eff_vol_addon: float
    eff_unhedged_penalty: float
    eff_floor_adjustment: float
    gross_market_value: float
    haircut_dollars: float
    eligible_collateral: float
    hedged_notional: float


class ReceivableValuation(BaseModel):
    buyer_name: str
    commodity: str
    gross_amount: float
    days_past_due: int
    is_investment_grade: bool
    is_aging_eligible: bool
    aging_ineligible_amount: float
    eligible_gross_amount: float
    ar_advance_rate: float
    ar_haircut_amount: float
    pre_cap_eligible: float
    cap_dollar_limit: float
    final_eligible: float
    concentration_cutback: float
    is_capped: bool
    pct_of_bb: float


class WaterfallLine(BaseModel):
    label: str
    amount: float
    measure: Literal["absolute", "relative", "total"]
    running_total: float


class BorrowingBaseResult(BaseModel):
    inventory_items: list[InventoryValuation]
    receivable_items: list[ReceivableValuation]
    waterfall: list[WaterfallLine]
    gross_inventory: float
    inventory_haircut_dollars: float
    eligible_inventory: float
    weighted_inv_advance_rate: float
    total_hedged_notional: float
    weighted_hedge_ratio: float
    gross_receivables: float
    aging_ineligible_dollars: float
    ar_advance_haircut_dollars: float
    pre_cap_eligible_receivables: float
    concentration_cutback_dollars: float
    final_eligible_receivables: float
    concentration_cap_pct: float
    cap_dollar_limit: float
    capped_buyer_names: list[str]
    solver_iterations: int
    solver_used_closed_form: bool
    solver_residual: float
    final_borrowing_base: float
    facility_limit: float
    drawn_revolver: float
    available_headroom: float
    facility_utilization_pct: float
    collateral_coverage_ratio: float


class CapTierComparison(BaseModel):
    cap_pct: float
    final_borrowing_base: float
    cap_dollar_limit: float
    total_cutback_dollars: float
    capped_buyer_count: int
    capped_buyer_names: list[str]
    available_headroom: float
    solver_iterations: int


class StressScenarioResult(BaseModel):
    shock_pct: float
    shocks_map: dict[str, float]
    baseline_bb: float
    shocked_bb: float
    bb_change: float
    baseline_headroom: float
    shocked_headroom: float
    headroom_change: float
    futures_mtm_pnl: float
    vm_outflow: float
    vm_inflow: float
    im_addon_outflow: float
    bb_deficiency_cure: float
    total_cash_outflows: float
    total_liquid_resources: float
    net_liquidity_buffer: float
    regime: LiquidityRegime
    regime_explanation: str
    shocked_bb_result: BorrowingBaseResult


class StressSweepResult(BaseModel):
    points: list[StressScenarioResult]
    upside_squeeze_breakeven_pct: float | None
    downside_deficiency_breakeven_pct: float | None
    worst_net_buffer_point: StressScenarioResult
    regime_counts: dict[str, int]

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "shock_pct": p.shock_pct,
                "baseline_bb": p.baseline_bb,
                "shocked_bb": p.shocked_bb,
                "bb_change": p.bb_change,
                "baseline_headroom": p.baseline_headroom,
                "shocked_headroom": p.shocked_headroom,
                "headroom_change": p.headroom_change,
                "futures_mtm_pnl": p.futures_mtm_pnl,
                "vm_outflow": p.vm_outflow,
                "vm_inflow": p.vm_inflow,
                "im_addon_outflow": p.im_addon_outflow,
                "bb_deficiency_cure": p.bb_deficiency_cure,
                "total_cash_outflows": p.total_cash_outflows,
                "total_liquid_resources": p.total_liquid_resources,
                "net_liquidity_buffer": p.net_liquidity_buffer,
                "regime": str(p.regime),
            }
            for p in self.points
        ]
        return pd.DataFrame(rows)


COMMODITY_SPECS: dict[str, CommoditySpec] = {
    "GC=F": CommoditySpec(
        ticker="GC=F", name="Gold", exchange="COMEX", unit="troy oz",
        contract_multiplier=100.0, base_haircut=0.14, ref_vol=0.15, vol_slope=0.50,
        unhedged_penalty_rate=0.15, min_haircut=0.10, max_haircut=0.65,
    ),
    "HG=F": CommoditySpec(
        ticker="HG=F", name="Copper", exchange="COMEX", unit="lbs",
        contract_multiplier=25000.0, base_haircut=0.18, ref_vol=0.20, vol_slope=0.50,
        unhedged_penalty_rate=0.15, min_haircut=0.10, max_haircut=0.65,
    ),
    "LBR=F": CommoditySpec(
        ticker="LBR=F", name="Lumber", exchange="CME", unit="mbf",
        contract_multiplier=27.5, base_haircut=0.24, ref_vol=0.30, vol_slope=0.45,
        unhedged_penalty_rate=0.18, min_haircut=0.10, max_haircut=0.65,
    ),
    "CC=F": CommoditySpec(
        ticker="CC=F", name="Cocoa", exchange="ICE", unit="MT",
        contract_multiplier=10.0, base_haircut=0.28, ref_vol=0.30, vol_slope=0.45,
        unhedged_penalty_rate=0.18, min_haircut=0.10, max_haircut=0.65,
    ),
}

DEFAULT_INVENTORY_POSITIONS: tuple[InventoryPosition, ...] = (
    InventoryPosition(ticker="GC=F", physical_qty=9_000.0, hedged_qty=9_000.0),
    InventoryPosition(ticker="HG=F", physical_qty=5_500_000.0, hedged_qty=4_675_000.0),
    InventoryPosition(ticker="LBR=F", physical_qty=32_000.0, hedged_qty=19_200.0),
    InventoryPosition(ticker="CC=F", physical_qty=2_900.0, hedged_qty=2_610.0),
)

DEFAULT_RECEIVABLES: tuple[ReceivableItem, ...] = (
    ReceivableItem(buyer_name="Barry Callebaut AG", commodity="Cocoa",
                   gross_amount=25_000_000.0, days_past_due=18, is_investment_grade=True),
    ReceivableItem(buyer_name="Cargill Cocoa & Chocolate", commodity="Cocoa",
                   gross_amount=21_500_000.0, days_past_due=24, is_investment_grade=True),
    ReceivableItem(buyer_name="Olam Food Ingredients", commodity="Cocoa",
                   gross_amount=8_500_000.0, days_past_due=32, is_investment_grade=True),
    ReceivableItem(buyer_name="West Coast Lumber Supply", commodity="Lumber",
                   gross_amount=5_200_000.0, days_past_due=41, is_investment_grade=False),
    ReceivableItem(buyer_name="Midwest Copper Fabricators", commodity="Copper",
                   gross_amount=4_100_000.0, days_past_due=14, is_investment_grade=False),
    ReceivableItem(buyer_name="Artisanal Roasters Co.", commodity="Cocoa",
                   gross_amount=2_400_000.0, days_past_due=104, is_investment_grade=False),
)

DEFAULT_FACILITY_TERMS = FacilityTerms()

STRESS_SCENARIO_PRESETS: dict[str, dict[str, float]] = {
    "Parallel +14% Margin Squeeze (Default)": {"GC=F": 0.14, "HG=F": 0.14, "LBR=F": 0.14, "CC=F": 0.14},
    "Baseline (0% Shock)": {"GC=F": 0.0, "HG=F": 0.0, "LBR=F": 0.0, "CC=F": 0.0},
    "West Africa Cocoa Spike (+30% CC, +5% Others)": {"GC=F": 0.05, "HG=F": 0.05, "LBR=F": 0.05, "CC=F": 0.30},
    "Broad Deflation Crash (-20% All)": {"GC=F": -0.20, "HG=F": -0.20, "LBR=F": -0.20, "CC=F": -0.20},
    "Metals Rally / Lumber Slump (+20% GC/HG, -10% LBR)": {"GC=F": 0.20, "HG=F": 0.20, "LBR=F": -0.10, "CC=F": 0.0},
}


METALS_TICKERS: tuple[str, ...] = ("GC=F", "HG=F")
SOFTS_TICKERS: tuple[str, ...] = ("LBR=F", "CC=F")


class ShockGridResult(BaseModel):
    metals_axis: list[float]
    softs_axis: list[float]
    net_buffer: list[list[float]]
    shocked_bb: list[list[float]]
    regime: list[list[str]]
    squeeze_share: float
    worst_net_buffer: float
    worst_metals_shock: float
    worst_softs_shock: float


class MonteCarloResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    horizon_days: int
    n_sims: int
    seed: int
    lookback_days: int
    net_buffer_samples: list[float]
    mean_net_buffer: float
    median_net_buffer: float
    p05_net_buffer: float
    p01_net_buffer: float
    es95_net_buffer: float
    prob_critical: float
    prob_margin_trap: float
    prob_deficiency: float
    prob_healthy: float
    regime_counts: dict[str, int]
    horizon_vols: dict[str, float]
    correlation: dict[str, dict[str, float]]
    worst_scenario_shocks: dict[str, float]


class HistoricalReplayResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame: pd.DataFrame
    start_date: str
    end_date: str
    n_days: int
    min_headroom: float
    min_headroom_date: str
    max_5d_vm_outflow: float
    max_5d_vm_outflow_date: str
    deficiency_days: int
    min_stressed_buffer: float
    min_stressed_buffer_date: str
