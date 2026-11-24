"""Typed inputs for the daily treasury engine (`core/treasury_engine.py`).

The engine is instrument-agnostic: instruments are identified by string ids and every array is indexed in the
order of `TreasuryBook.instruments`. Money is in the book's functional currency (one currency per run; convert
upstream with `core.units.convert_currency`). Quantities are in each instrument's physical unit.

Assumption labels (`Provenance`) are carried on every parameter group so reports can separate:
- "contractual": a term of the (hypothetical) financing agreement or exchange rulebook entered by the user
- "calibrated": estimated from the loaded price history by this application
- "illustrative": a default chosen for demonstration; not market, bank or exchange data
- "synthetic_stress": a scenario assumption, not a forecast
- "user": edited by the user in the interface
None of the defaults in this module is sourced from a real facility agreement, exchange notice or rating agency.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from core.units import UNITS

Provenance = Literal["contractual", "calibrated", "illustrative", "synthetic_stress", "user", "observed"]
HaircutConvention = Literal["haircut_only", "haircut_and_advance"]
MarginMode = Literal["fixed_per_contract", "pct_notional", "vol_scaled"]
RATING_SCALE: tuple[str, ...] = ("AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D")
INVESTMENT_GRADE = frozenset({"AAA", "AA", "A", "BBB"})

# Illustrative one-year default probabilities by rating bucket. These are round numbers chosen to be of a
# plausible order of magnitude for demonstration; they are not taken from any rating agency study.
ILLUSTRATIVE_ANNUAL_PD: dict[str, float] = {
    "AAA": 0.0002, "AA": 0.0005, "A": 0.001, "BBB": 0.003, "BB": 0.012, "B": 0.04, "CCC": 0.20, "D": 1.0,
}


class Instrument(BaseModel):
    id: str = Field(min_length=1)
    name: str
    unit: str
    contract_size: float = Field(gt=0.0, description="physical units per futures contract")
    currency: str = "USD"
    exchange: str = ""

    @field_validator("unit")
    @classmethod
    def _unit(cls, v: str) -> str:
        if v not in UNITS:
            raise ValueError(f"unknown unit {v!r}")
        return v


class HaircutPolicy(BaseModel):
    """Inventory valuation haircut, applied as

        h(t) = clip(base + vol_coefficient * max(0, vol(t) - reference_vol) + liquidity + location + quality
                    + age_per_30d * max(0, age_days - age_free_days) / 30 + concentration (if share > threshold)
                    + unhedged_penalty * (1 - hedge ratio) + lender_stress(t), min_haircut, max_haircut)

    If `dynamic` is False, vol(t) is frozen at the opening volatility and lender stress at its opening value,
    so the haircut is static. `max_increase_per_redetermination` models a contractual limit on how fast the
    lender may raise the haircut (None = no limit). This is an illustrative specification, not a regulatory or
    industry-standard formula."""

    base: float = Field(default=0.20, ge=0.0, le=0.95)
    vol_coefficient: float = Field(default=0.5, ge=0.0, le=5.0)
    reference_vol: float = Field(default=0.25, ge=0.0, le=5.0)
    liquidity_addon: float = Field(default=0.0, ge=0.0, le=0.5)
    location_addon: float = Field(default=0.0, ge=0.0, le=0.5)
    quality_addon: float = Field(default=0.0, ge=0.0, le=0.5)
    age_addon_per_30d: float = Field(default=0.0, ge=0.0, le=0.5)
    age_free_days: int = Field(default=90, ge=0, le=3650)
    concentration_threshold: float = Field(default=1.0, gt=0.0, le=1.0)
    concentration_addon: float = Field(default=0.0, ge=0.0, le=0.5)
    unhedged_penalty: float = Field(default=0.0, ge=0.0, le=0.5)
    min_haircut: float = Field(default=0.10, ge=0.0, le=1.0)
    max_haircut: float = Field(default=0.65, ge=0.0, le=1.0)
    dynamic: bool = True
    max_increase_per_redetermination: float | None = Field(default=None, ge=0.0, le=1.0)
    provenance: Provenance = "illustrative"

    @model_validator(mode="after")
    def _bounds(self) -> HaircutPolicy:
        if self.min_haircut > self.max_haircut:
            raise ValueError("min_haircut cannot exceed max_haircut")
        return self


class MarginPolicy(BaseModel):
    """Initial margin per contract. All modes are proxies, not an exchange's clearing model (SPAN, IRM or VaR):

    - fixed_per_contract: im_per_contract (a schedule value entered by the user)
    - pct_notional: pct_notional x futures price x contract size
    - vol_scaled: z x vol(t) x sqrt(mpor_days / 252) x futures price x contract size

    The rate is reset every `review_days` business days (exchanges change margins periodically, not
    continuously), never falls below `floor_fraction` x the opening rate (an anti-procyclicality floor), and is
    multiplied by the scenario step-up path. Excess margin is returned `release_lag_days` after it falls due."""

    mode: MarginMode = "pct_notional"
    im_per_contract: float = Field(default=0.0, ge=0.0, le=1e8)
    pct_notional: float = Field(default=0.10, ge=0.0, le=1.0)
    z_score: float = Field(default=2.33, ge=0.0, le=6.0)
    mpor_days: int = Field(default=2, ge=1, le=20)
    floor_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    review_days: int = Field(default=1, ge=1, le=60)
    release_lag_days: int = Field(default=0, ge=0, le=20)
    stepup_trigger: float | None = Field(default=None, ge=0.0, le=5.0)
    stepup_fraction: float = Field(default=0.0, ge=0.0, le=5.0)
    provenance: Provenance = "illustrative"


class PhysicalPosition(BaseModel):
    instrument_id: str
    quantity: float = Field(ge=0.0)
    purchase_price: float | None = Field(default=None, ge=0.0)
    opening_basis_pct: float = Field(default=0.0, ge=-0.9, le=2.0)
    location: str = ""
    quality: str = ""
    age_days: int = Field(default=0, ge=0)
    financing_eligible: bool = True
    in_transit_qty: float = Field(default=0.0, ge=0.0)
    transit_days_remaining: int = Field(default=0, ge=0)
    storage_cost_per_unit_day: float = Field(default=0.0, ge=0.0)
    hedge_ratio_target: float | None = Field(default=None, ge=0.0, le=2.0)


class FuturesPosition(BaseModel):
    """Signed contracts: negative = short. `contract_index` 0 is the nearest contract in the curve arrays."""

    instrument_id: str
    contract_id: str = ""
    contract_index: int = Field(default=0, ge=0)
    signed_contracts: float
    entry_price: float | None = None
    hedges_physical: bool = True

    @property
    def is_integer(self) -> bool:
        return abs(self.signed_contracts - round(self.signed_contracts)) < 1e-9


class Counterparty(BaseModel):
    id: str = Field(min_length=1)
    name: str
    industry: str = ""
    region: str = ""
    rating: str = "BBB"
    annual_pd: float | None = Field(default=None, ge=0.0, le=1.0)
    lgd: float = Field(default=0.6, ge=0.0, le=1.0)
    credit_limit: float | None = Field(default=None, ge=0.0)
    insurance_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    insurance_deductible: float = Field(default=0.0, ge=0.0)
    insurance_limit: float | None = Field(default=None, ge=0.0)
    insurance_excluded: bool = False
    claim_delay_days: int = Field(default=180, ge=0, le=720)
    recovery_delay_days: int = Field(default=365, ge=0, le=3650)
    expected_delay_days: int = Field(default=0, ge=0, le=365)
    commodity_loading: float = Field(default=0.0, ge=-1.0, le=1.0)
    commodity_link: str | None = None
    provenance: Provenance = "illustrative"

    @field_validator("rating")
    @classmethod
    def _rating(cls, v: str) -> str:
        if v not in RATING_SCALE:
            raise ValueError(f"rating must be one of {RATING_SCALE}")
        return v

    @property
    def pd_annual(self) -> float:
        return ILLUSTRATIVE_ANNUAL_PD[self.rating] if self.annual_pd is None else self.annual_pd

    @property
    def is_investment_grade(self) -> bool:
        return self.rating in INVESTMENT_GRADE


class Invoice(BaseModel):
    """An opening receivable. `due_in_days` is calendar days from the start date until contractual due date
    (0 when already due); `days_past_due` is the age past due at the start."""

    counterparty_id: str
    amount: float = Field(ge=0.0)
    days_past_due: int = Field(default=0, ge=0)
    due_in_days: int = Field(default=0, ge=0)
    commodity: str = ""


class Payable(BaseModel):
    amount: float = Field(ge=0.0)
    due_in_days: int = Field(default=0, ge=0)
    supplier: str = ""


class PhysicalTrade(BaseModel):
    """A scheduled physical purchase or sale on business day `day` (1-based) of the simulation.

    Price = physical market price on the trade day x (1 + price_premium_pct) when price_mode is "market",
    or `fixed_price`. Purchases: title passes on the trade day (inventory in transit for `transit_days`
    business days), cash is paid `payment_days` calendar days later (a payable meanwhile). Sales: inventory
    leaves on the trade day and an invoice to `counterparty_id` falls due `payment_days` calendar days later."""

    day: int = Field(ge=1)
    instrument_id: str
    side: Literal["buy", "sell"]
    quantity: float = Field(gt=0.0)
    price_mode: Literal["market", "fixed"] = "market"
    fixed_price: float | None = Field(default=None, ge=0.0)
    price_premium_pct: float = Field(default=0.0, ge=-0.5, le=0.5)
    payment_days: int = Field(default=30, ge=0, le=365)
    transit_days: int = Field(default=0, ge=0, le=120)
    counterparty_id: str | None = None

    @model_validator(mode="after")
    def _check(self) -> PhysicalTrade:
        if self.price_mode == "fixed" and self.fixed_price is None:
            raise ValueError("fixed price trades need fixed_price")
        if self.side == "sell" and not self.counterparty_id:
            raise ValueError("a sale needs a counterparty_id")
        return self


class LiquiditySource(BaseModel):
    """A committed or emergency line used after the revolver in the funding waterfall."""

    name: str
    limit: float = Field(ge=0.0)
    delay_days: int = Field(default=0, ge=0, le=20)
    interest_rate: float = Field(default=0.08, ge=0.0, le=1.0)
    margin_only: bool = False
    repay_from_margin_receipts: bool = True
    enabled: bool = True
    provenance: Provenance = "illustrative"


class FacilityConfig(BaseModel):
    """Revolving borrowing base facility. Borrowing base:

        inventory contribution = sum eligible MV x (1 - haircut) [x inventory_advance_rate if convention is
                                  haircut_and_advance], capped by inventory_sublimit
        receivable contribution = sum over counterparties of min(pre-cap eligible, cap x BB), where pre-cap
                                  eligible = (eligible face - dilution reserve) x advance rate; capped by
                                  receivable_sublimit
        BB = inventory + receivables - fixed reserves (floored at zero)
        permitted = min(reported BB, commitment); undrawn = max(0, permitted - drawn); deficiency = max(0,
        drawn - permitted)."""

    commitment: float = Field(gt=0.0)
    opening_drawn: float = Field(ge=0.0)
    interest_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    commitment_fee: float = Field(default=0.0, ge=0.0, le=0.1)
    convention: HaircutConvention = "haircut_only"
    inventory_advance_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    in_transit_eligible: bool = False
    ar_ig_advance_rate: float = Field(default=0.85, ge=0.0, le=1.0)
    ar_non_ig_advance_rate: float = Field(default=0.75, ge=0.0, le=1.0)
    insured_ar_advance_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    ar_aging_limit_days: int = Field(default=90, ge=0)
    min_eligible_rating: str = "CCC"
    concentration_cap_pct: float = Field(default=1.0, gt=0.0, le=1.0)
    dilution_reserve_pct: float = Field(default=0.0, ge=0.0, le=0.5)
    inventory_sublimit: float | None = Field(default=None, ge=0.0)
    receivable_sublimit: float | None = Field(default=None, ge=0.0)
    fixed_reserves: float = Field(default=0.0, ge=0.0)
    drawdown_delay_days: int = Field(default=1, ge=0, le=10)
    redetermination_days: int = Field(default=1, ge=1, le=60)
    cure_grace_days: int = Field(default=2, ge=0, le=30)
    sweep_surplus: bool = True
    provenance: Provenance = "illustrative"

    @field_validator("min_eligible_rating")
    @classmethod
    def _rating(cls, v: str) -> str:
        if v not in RATING_SCALE:
            raise ValueError(f"rating must be one of {RATING_SCALE}")
        return v


class TreasuryConfig(BaseModel):
    start_date: date = date(2026, 11, 20)
    horizon_days: int = Field(default=30, ge=1, le=260)
    opening_cash: float = Field(ge=0.0)
    other_restricted_cash: float = Field(default=0.0, ge=0.0)
    min_operating_cash: float = Field(default=0.0, ge=0.0)
    sweep_target_cash: float | None = Field(default=None, ge=0.0)
    opex_per_calendar_day: float = Field(default=0.0, ge=0.0)
    day_count: Literal[360, 365] = 360
    facility: FacilityConfig
    extra_lines: list[LiquiditySource] = Field(default_factory=list)
    emergency: LiquiditySource | None = None
    haircuts: dict[str, HaircutPolicy]
    margins: dict[str, MarginPolicy]
    transaction_cost_per_contract: float = Field(default=0.0, ge=0.0)
    hedge_follows_inventory: bool = False
    roll_days_before_expiry: int = Field(default=2, ge=0, le=20)


class TreasuryBook(BaseModel):
    instruments: list[Instrument]
    physical: list[PhysicalPosition] = Field(default_factory=list)
    futures: list[FuturesPosition] = Field(default_factory=list)
    counterparties: list[Counterparty] = Field(default_factory=list)
    invoices: list[Invoice] = Field(default_factory=list)
    payables: list[Payable] = Field(default_factory=list)
    trades: list[PhysicalTrade] = Field(default_factory=list)

    @model_validator(mode="after")
    def _refs(self) -> TreasuryBook:
        ids = [i.id for i in self.instruments]
        if len(set(ids)) != len(ids):
            raise ValueError("instrument ids must be unique")
        cps = {c.id for c in self.counterparties}
        if len(cps) != len(self.counterparties):
            raise ValueError("counterparty ids must be unique")
        for obj in [*self.physical, *self.futures, *self.trades]:
            if obj.instrument_id not in ids:
                raise ValueError(f"unknown instrument {obj.instrument_id!r}")
        for inv in self.invoices:
            if inv.counterparty_id not in cps:
                raise ValueError(f"unknown counterparty {inv.counterparty_id!r}")
        for tr in self.trades:
            if tr.side == "sell" and tr.counterparty_id not in cps:
                raise ValueError(f"unknown counterparty {tr.counterparty_id!r}")
        return self

    def index(self, instrument_id: str) -> int:
        return [i.id for i in self.instruments].index(instrument_id)
