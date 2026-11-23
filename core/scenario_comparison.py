"""Side-by-side scenario comparison and versioned, validated workbench configurations (JSON).

Comparison columns are computed with one engine and one set of conventions per call: in legacy mode the
advanced scenario runs with neutral settings (identical to the legacy stress engine) and advanced-only
metrics are reported as not applicable (None).
"""

from __future__ import annotations

import base64
import binascii
import json
import zlib
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from core.advanced_scenario import auto_im_multipliers, neutral_settings, run_advanced_scenario
from core.advanced_schemas import (
    AdvancedSettings,
    BasisAssumption,
    CashflowSettings,
    MarginSettings,
    ScenarioSpec,
    SimulationMethod,
)
from core.cashflow_engine import deterministic_paths, run_cashflow_ledger
from core.schemas import (
    COMMODITY_SPECS,
    REQUIRED_TICKERS,
    CommoditySpec,
    FacilityTerms,
    InventoryPosition,
    MarketQuote,
    ReceivableItem,
)

ModelMode = Literal["legacy", "advanced"]

COMPARISON_METRICS: tuple[tuple[str, str], ...] = (
    ("physical_inventory_value", "Physical inventory value"),
    ("borrowing_base", "Borrowing base"),
    ("revolver_availability", "Revolver availability"),
    ("im_requirement", "Initial margin requirement"),
    ("vm_net", "Variation margin (net, + = received)"),
    ("margin_line_draw", "Margin line draw"),
    ("bb_cure", "Borrowing base deficiency cure"),
    ("net_liquidity_buffer", "Net liquidity buffer"),
    ("combined_pnl", "Combined economic P&L"),
    ("funding_shortfall", "Largest unfunded daily gap (ledger)"),
)


class ComparisonColumn(BaseModel):
    label: str
    regime: str
    metrics: dict[str, float | None]


def compare_scenarios(
    positions: Sequence[InventoryPosition],
    receivables: Sequence[ReceivableItem],
    market_snapshot: Mapping[str, MarketQuote],
    terms: FacilityTerms,
    settings: AdvancedSettings,
    mode: ModelMode,
    scenarios: Sequence[ScenarioSpec],
    cashflow: CashflowSettings | None = None,
    apply_trigger_rule: bool = True,
    specs: Mapping[str, CommoditySpec] = COMMODITY_SPECS,
) -> list[ComparisonColumn]:
    """Run each scenario. In advanced mode `im_multipliers` are the user's stress multipliers; the per-commodity
    step-up trigger rule is applied on top (instantaneous view and ledger alike)."""
    use = settings if mode == "advanced" else neutral_settings(settings)
    cols = []
    for spec in scenarios:
        run_spec = spec if mode == "advanced" else ScenarioSpec(name=spec.name, futures_shocks=spec.futures_shocks)
        instant = run_spec
        if mode == "advanced" and use.margin.im_mode == "per_contract":
            instant = ScenarioSpec.model_validate({**run_spec.model_dump(), "im_multipliers": auto_im_multipliers(
                run_spec.futures_shocks, terms, run_spec.im_multipliers, apply_trigger_rule)})
        r = run_advanced_scenario(positions, receivables, market_snapshot, terms, use, instant, specs)
        shortfall: float | None = None
        if mode == "advanced" and cashflow is not None:
            fp, bp = deterministic_paths(market_snapshot, use, run_spec, cashflow)
            cf = CashflowSettings.model_validate({**cashflow.model_dump(), "margin_line_cut_pct": max(
                cashflow.margin_line_cut_pct, run_spec.margin_line_cut_pct)})
            run = run_cashflow_ledger(positions, receivables, market_snapshot, terms, use, cf, fp, bp,
                                      run_spec.im_multipliers, apply_trigger_rule, specs)
            shortfall = float(run.max_shortfall[0])
        cols.append(ComparisonColumn(label=spec.name, regime=str(r.regime), metrics={
            "physical_inventory_value": r.physical_inventory_value_1,
            "borrowing_base": r.shocked_bb,
            "revolver_availability": r.shocked_headroom,
            "im_requirement": r.im_required,
            "vm_net": r.vm_net,
            "margin_line_draw": r.margin_line_draw,
            "bb_cure": r.bb_deficiency_cure,
            "net_liquidity_buffer": r.net_liquidity_buffer,
            "combined_pnl": r.combined_pnl,
            "funding_shortfall": shortfall,
        }))
    return cols


# ---- Saved configurations -----------------------------------------------------------------------------------

CONFIG_VERSION = 1
MAX_CONFIG_BYTES = 1_000_000


class ConfigError(ValueError):
    """Raised when an uploaded configuration cannot be used; the message is safe to show to the user."""


class ScenarioState(BaseModel):
    parallel_shock_pct: int = Field(ge=-20, le=20)
    override_enabled: bool = False
    per_commodity_pct: dict[str, int] = Field(default_factory=dict)

    @field_validator("per_commodity_pct")
    @classmethod
    def _pc(cls, v: dict[str, int]) -> dict[str, int]:
        for t, x in v.items():
            if t not in REQUIRED_TICKERS or not -30 <= int(x) <= 30:
                raise ValueError(f"invalid per-commodity shock {t}: {x}")
        return v


class SimulationState(BaseModel):
    mc_horizon: Literal[1, 5, 10, 20] = 5
    mc_sims: Literal[1000, 2500, 5000, 10000] = 5000
    calibration_window: Literal["60 days", "125 days", "250 days", "Maximum"] = "250 days"
    method: SimulationMethod = "gaussian"
    basis_vol_annual: float = Field(default=0.0, ge=0.0, le=1.0)


class WorkbenchConfig(BaseModel):
    """Versioned, data-only snapshot of every user input. Loading validates every field; nothing is executed."""

    kind: Literal["commodity-liquidity-workbench"] = "commodity-liquidity-workbench"
    version: int = CONFIG_VERSION
    model_mode: ModelMode = "legacy"
    positions: list[InventoryPosition]
    receivables: list[ReceivableItem]
    terms: FacilityTerms
    margin: MarginSettings
    basis: dict[str, BasisAssumption]
    scenario_extras: ScenarioSpec = Field(default_factory=ScenarioSpec)
    apply_trigger_rule: bool = True
    cashflow: CashflowSettings = Field(default_factory=CashflowSettings)
    scenario: ScenarioState
    simulation: SimulationState = Field(default_factory=SimulationState)
    report_ccy: Literal["USD", "EUR", "CHF", "GBP"] = "USD"

    @field_validator("positions")
    @classmethod
    def _positions(cls, v: list[InventoryPosition]) -> list[InventoryPosition]:
        if sorted(p.ticker for p in v) != sorted(REQUIRED_TICKERS):
            raise ValueError("positions must contain exactly one entry for each of the six commodities")
        return v


def dump_config(config: WorkbenchConfig) -> str:
    return json.dumps(config.model_dump(mode="json"), indent=2)


def load_config(raw: str | bytes) -> WorkbenchConfig:
    """Parse and validate an uploaded configuration. Raises ConfigError with a readable reason."""
    if isinstance(raw, bytes):
        if len(raw) > MAX_CONFIG_BYTES:
            raise ConfigError("File is larger than 1 MB; this is not a workbench configuration.")
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConfigError("File is not UTF-8 text.") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"File is not valid JSON (line {exc.lineno}).") from exc
    if not isinstance(data, dict) or data.get("kind") != "commodity-liquidity-workbench":
        raise ConfigError("File is not a Commodity Liquidity Workbench configuration.")
    if data.get("version") != CONFIG_VERSION:
        raise ConfigError(f"Unsupported configuration version {data.get('version')!r}; expected {CONFIG_VERSION}.")
    try:
        return WorkbenchConfig.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(x) for x in first["loc"])
        raise ConfigError(f"Invalid value at {where}: {first['msg']}.") from exc


MAX_SHARE_TOKEN_CHARS = 24_000


def encode_share_token(config_json: str) -> str:
    """Compact, URL-safe token for a configuration: minified JSON, zlib level 9, base64url without padding."""
    minified = json.dumps(json.loads(config_json), separators=(",", ":"), sort_keys=True)
    return base64.urlsafe_b64encode(zlib.compress(minified.encode("utf-8"), 9)).decode("ascii").rstrip("=")


def decode_share_token(token: str) -> WorkbenchConfig:
    """Inverse of encode_share_token. Bounded in size and validated through load_config; raises ConfigError."""
    token = token.strip()
    if not token or len(token) > MAX_SHARE_TOKEN_CHARS:
        raise ConfigError("The shared link is empty or too long to be a workbench scenario.")
    try:
        compressed = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
        inflater = zlib.decompressobj()
        raw = inflater.decompress(compressed, MAX_CONFIG_BYTES)
        if inflater.unconsumed_tail:
            raise ConfigError("The shared link expands beyond 1 MB; this is not a workbench scenario.")
    except (binascii.Error, ValueError, zlib.error) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError("The shared link is damaged or incomplete.") from exc
    return load_config(raw)
