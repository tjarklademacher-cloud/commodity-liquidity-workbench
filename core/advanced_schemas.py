"""Validated inputs and outputs for the advanced model (margin engine, basis risk, daily cash flows).

The legacy model in `schemas.py` is untouched. Every advanced input has a neutral setting under which the
advanced instantaneous scenario reproduces the legacy stress result exactly (see `advanced_scenario.py`).
All default values are illustrative assumptions, not exchange, bank or market data.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from core.schemas import REQUIRED_TICKERS, BorrowingBaseResult, LiquidityRegime

AssumptionType = Literal["illustrative", "user", "observed"]
ContractPolicy = Literal["continuous", "nearest", "floor", "ceil"]
IMMode = Literal["legacy_stepup", "per_contract"]
SimulationMethod = Literal["gaussian", "bootstrap", "stress"]


def _check_ticker(value: str) -> str:
    if value not in REQUIRED_TICKERS:
        raise ValueError(f"ticker must be one of {REQUIRED_TICKERS}, got {value!r}")
    return value


def _check_map(values: dict[str, float], lo: float, hi: float, name: str) -> dict[str, float]:
    for t, v in values.items():
        _check_ticker(t)
        if not (lo <= float(v) <= hi):
            raise ValueError(f"{name} for {t} must be between {lo} and {hi}, got {v}")
    return {t: float(v) for t, v in values.items()}


class MarginProfile(BaseModel):
    """Per-commodity futures margin assumption. Amounts are USD per contract."""

    ticker: str
    im_per_contract: float = Field(ge=0.0, le=1e7)
    maintenance_per_contract: float | None = Field(default=None, ge=0.0, le=1e7)
    fixed_addon_usd: float = Field(default=0.0, ge=0.0, le=1e9)
    financeable: bool = True
    source: str = "Illustrative assumption"
    source_date: str = ""
    assumption_type: AssumptionType = "illustrative"

    @field_validator("ticker")
    @classmethod
    def _ticker(cls, v: str) -> str:
        return _check_ticker(v)

    @model_validator(mode="after")
    def _maintenance_not_above_initial(self) -> MarginProfile:
        if self.maintenance_per_contract is not None and self.maintenance_per_contract > self.im_per_contract:
            raise ValueError("maintenance margin cannot exceed initial margin")
        return self


class MarginSettings(BaseModel):
    profiles: dict[str, MarginProfile]
    contract_policy: ContractPolicy = "nearest"
    im_mode: IMMode = "per_contract"
    release_excess_im: bool = True

    @model_validator(mode="after")
    def _all_tickers(self) -> MarginSettings:
        missing = [t for t in REQUIRED_TICKERS if t not in self.profiles]
        if missing:
            raise ValueError(f"margin profiles missing for {missing}")
        return self


class BasisAssumption(BaseModel):
    """Structural basis for one commodity. Basis = physical price - futures price, expressed as a share of
    the futures price, so physical = futures * (1 + basis)."""

    ticker: str
    opening_basis_pct: float = Field(default=0.0, ge=-0.5, le=0.5)

    @field_validator("ticker")
    @classmethod
    def _ticker(cls, v: str) -> str:
        return _check_ticker(v)


class ScenarioSpec(BaseModel):
    """One instantaneous market scenario for the advanced model. Missing tickers mean no change."""

    name: str = "Scenario"
    futures_shocks: dict[str, float] = Field(default_factory=dict)
    im_multipliers: dict[str, float] = Field(default_factory=dict)
    basis_shocks: dict[str, float] = Field(default_factory=dict)
    liquidation_discounts: dict[str, float] = Field(default_factory=dict)
    margin_line_cut_pct: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("futures_shocks")
    @classmethod
    def _fs(cls, v: dict[str, float]) -> dict[str, float]:
        return _check_map(v, -0.95, 3.0, "futures shock")

    @field_validator("im_multipliers")
    @classmethod
    def _im(cls, v: dict[str, float]) -> dict[str, float]:
        return _check_map(v, 0.0, 10.0, "IM multiplier")

    @field_validator("basis_shocks")
    @classmethod
    def _bs(cls, v: dict[str, float]) -> dict[str, float]:
        return _check_map(v, -0.5, 0.5, "basis shock")

    @field_validator("liquidation_discounts")
    @classmethod
    def _ld(cls, v: dict[str, float]) -> dict[str, float]:
        return _check_map(v, 0.0, 0.5, "liquidation discount")


class AdvancedSettings(BaseModel):
    margin: MarginSettings
    basis: dict[str, BasisAssumption]

    @model_validator(mode="after")
    def _all_basis(self) -> AdvancedSettings:
        missing = [t for t in REQUIRED_TICKERS if t not in self.basis]
        if missing:
            raise ValueError(f"basis assumptions missing for {missing}")
        return self


class CommodityMarginLine(BaseModel):
    """Per-commodity audit of the advanced scenario. Prices are USD per physical unit."""

    ticker: str
    name: str
    physical_qty: float
    hedged_qty: float
    contracts_exact: float
    contracts: float
    residual_hedge_qty: float
    futures_price_0: float
    futures_price_1: float
    physical_price_0: float
    physical_price_1: float
    collateral_price_1: float
    futures_notional_1: float
    vm_cash: float
    im_per_contract: float
    im_multiplier: float
    im_required_0: float
    im_required_1: float
    financeable: bool
    physical_pnl: float
    futures_pnl: float
    basis_pnl: float
    residual_price_pnl: float
    liquidation_discount_cost: float


class ReconciliationStep(BaseModel):
    label: str
    net_liquidity_buffer: float
    change: float


class AdvancedScenarioResult(BaseModel):
    scenario: ScenarioSpec
    lines: list[CommodityMarginLine]
    baseline_bb: float
    shocked_bb_result: BorrowingBaseResult
    shocked_bb: float
    baseline_headroom: float
    shocked_headroom: float
    headroom_change: float
    vm_net: float
    vm_payable: float
    vm_receivable: float
    im_mode: IMMode
    im_posted_opening: float
    im_required: float
    im_additional: float
    im_released: float
    margin_cash_need: float
    margin_line_capacity: float
    margin_line_eligible_need: float
    margin_line_draw: float
    margin_funded_from_cash: float
    bb_deficiency_cure: float
    funding_carry_outflow: float = 0.0
    total_liquid_resources: float
    total_cash_outflows: float
    net_liquidity_buffer: float
    regime: LiquidityRegime
    regime_explanation: str
    physical_inventory_value_0: float
    physical_inventory_value_1: float
    physical_pnl: float
    futures_pnl: float
    combined_pnl: float
    basis_pnl: float
    residual_price_pnl: float
    liquidation_discount_cost: float


class CashflowSettings(BaseModel):
    """Daily treasury ledger conventions. Rates, delays and terms are illustrative defaults."""

    horizon_days: int = Field(default=20, ge=1, le=60)
    ramp_days: int = Field(default=5, ge=1, le=60)
    revolver_rate: float = Field(default=0.07, ge=0.0, le=0.5)
    margin_line_rate: float = Field(default=0.08, ge=0.0, le=0.5)
    day_count: Literal[360, 365] = 360
    revolver_delay_days: int = Field(default=1, ge=0, le=5)
    margin_line_delay_days: int = Field(default=0, ge=0, le=5)
    cure_grace_days: int = Field(default=2, ge=0, le=10)
    invoice_terms_days: int = Field(default=60, ge=1, le=180)
    overdue_collection_delay_days: int = Field(default=15, ge=0, le=180)
    noncollected_buyers: list[str] = Field(default_factory=list)
    repay_margin_line_from_receipts: bool = True
    sweep_surplus_to_revolver: bool = True
    minimum_cash: float = Field(default=0.0, ge=0.0, le=1e9)
    margin_line_cut_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    margin_line_cut_day: int = Field(default=1, ge=1, le=60)
