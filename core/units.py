"""Physical unit and currency conversion with explicit dimensions.

Quantities are converted only within one dimension (mass, volume, energy, board measure). Conversions across
dimensions (for example barrels to metric tonnes, or tonnes of coal to MWh) depend on the commodity's density
or energy content and are refused unless the caller supplies an explicit, documented factor through
`convert_with_factor`.

Exact factors used (definitions, not estimates):
- 1 troy ounce = 31.1034768 g; 1 lb = 0.45359237 kg (international avoirdupois pound, 1959)
- 1 short ton = 2,000 lb; 1 long ton = 2,240 lb; 1 metric tonne = 1,000 kg
- 1 US barrel = 42 US gallons = 158.987294928 litres; 1 cubic metre = 1,000 litres
- 1 MWh = 3.6 GJ; 1 therm = 105.5056 MJ (US therm); 1 MMBtu = 10 therms
- 1 mbf = 1,000 board feet
Bushels are a volume unit whose weight depends on the grain; converting bushels to tonnes needs a grain-specific
standard weight (for example 60 lb for US wheat and soybeans, 56 lb for corn), passed as a factor.
"""

from __future__ import annotations

from dataclasses import dataclass

MASS, VOLUME, ENERGY, BOARD = "mass", "volume", "energy", "board measure"


@dataclass(frozen=True)
class Unit:
    code: str
    dimension: str
    to_base: float  # base units: kg, litre, MJ, board foot


UNITS: dict[str, Unit] = {u.code: u for u in (
    Unit("kg", MASS, 1.0),
    Unit("g", MASS, 1e-3),
    Unit("MT", MASS, 1000.0),
    Unit("lbs", MASS, 0.45359237),
    Unit("troy oz", MASS, 0.0311034768),
    Unit("short ton", MASS, 2000 * 0.45359237),
    Unit("long ton", MASS, 2240 * 0.45359237),
    Unit("litre", VOLUME, 1.0),
    Unit("m3", VOLUME, 1000.0),
    Unit("bbl", VOLUME, 158.987294928),
    Unit("US gal", VOLUME, 3.785411784),
    Unit("bushel", VOLUME, 35.23907016688),
    Unit("MJ", ENERGY, 1.0),
    Unit("GJ", ENERGY, 1000.0),
    Unit("MWh", ENERGY, 3600.0),
    Unit("therm", ENERGY, 105.5056),
    Unit("MMBtu", ENERGY, 1055.056),
    Unit("board ft", BOARD, 1.0),
    Unit("mbf", BOARD, 1000.0),
)}


class UnitMismatchError(ValueError):
    """Raised when two units measure different physical dimensions."""


def unit(code: str) -> Unit:
    try:
        return UNITS[code]
    except KeyError as exc:
        raise ValueError(f"unknown unit {code!r}; known units: {sorted(UNITS)}") from exc


def conversion_factor(from_unit: str, to_unit: str) -> float:
    """Multiply a quantity in `from_unit` by this factor to express it in `to_unit`."""
    a, b = unit(from_unit), unit(to_unit)
    if a.dimension != b.dimension:
        raise UnitMismatchError(
            f"cannot convert {from_unit} ({a.dimension}) to {to_unit} ({b.dimension}) without a commodity-specific "
            "density or energy-content factor")
    return a.to_base / b.to_base


def convert_quantity(qty: float, from_unit: str, to_unit: str) -> float:
    return qty * conversion_factor(from_unit, to_unit)


def convert_price(price_per_from_unit: float, from_unit: str, to_unit: str) -> float:
    """A price per `from_unit` expressed per `to_unit` (prices convert inversely to quantities)."""
    return price_per_from_unit / conversion_factor(from_unit, to_unit)


def convert_with_factor(qty: float, factor: float, description: str) -> float:
    """Cross-dimension conversion with an explicit factor (for example 7.33 bbl per tonne for a given crude).
    The description is required so the assumption is recorded where it is used."""
    if factor <= 0 or not description.strip():
        raise ValueError("a positive factor and a description of its source are required")
    return qty * factor


def convert_currency(amount: float, from_ccy: str, to_ccy: str, usd_per_unit: dict[str, float]) -> float:
    """Convert through USD using quotes in USD per unit of each currency (USD itself is 1)."""
    rates = {"USD": 1.0, **usd_per_unit}
    for c in (from_ccy, to_ccy):
        if c not in rates or rates[c] <= 0:
            raise ValueError(f"no positive USD rate for {c}")
    return amount * rates[from_ccy] / rates[to_ccy]


def hedge_ratio_in_physical_units(physical_qty: float, physical_unit: str, contracts: float,
                                  contract_size: float, contract_unit: str) -> float:
    """Futures quantity (|contracts| x contract size, converted to the physical unit) over physical quantity."""
    if physical_qty <= 0:
        return 0.0
    futures_qty = convert_quantity(abs(contracts) * contract_size, contract_unit, physical_unit)
    return futures_qty / physical_qty
