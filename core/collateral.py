"""Dynamic collateral haircuts and the borrowing base, vectorized over simulation paths.

Haircut versus advance rate
- The valuation haircut reflects the risk of realizing the collateral (price volatility, market liquidity,
  location, quality, age, concentration, lender stress). Adjusted value = eligible MV x (1 - haircut).
- The advance rate is the contractual share of adjusted value the lender will lend.
- Convention "haircut_only" (the legacy model's convention): the haircut already is the complement of the
  advance rate, so inventory contribution = MV x (1 - haircut) and no second deduction is made.
- Convention "haircut_and_advance": contribution = MV x (1 - haircut) x inventory advance rate. Use it only when
  the agreement states both; `convention_warning` flags combinations that look like a double deduction.
Receivables use advance rates only (plus the dilution reserve), never an inventory haircut.

Each asset is pledged once: inventory and receivables are disjoint sets, a sold lot leaves inventory on the
trade day and enters receivables as an invoice, and in-transit stock is counted only if the facility says so.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from core.treasury_schemas import FacilityConfig, HaircutPolicy

HAIRCUT_COMPONENTS = ("base", "volatility", "liquidity", "location", "quality", "age", "concentration",
                      "unhedged", "lender_stress")


def haircut_components(
    policy: HaircutPolicy,
    vol: np.ndarray,
    inventory_share: np.ndarray,
    hedge_ratio: float,
    age_days: float,
    lender_stress: np.ndarray | float = 0.0,
) -> dict[str, np.ndarray]:
    """Uncapped components of the haircut for one instrument; arrays broadcast over paths."""
    vol = np.asarray(vol, dtype=float)
    ones = np.ones_like(vol)
    return {
        "base": policy.base * ones,
        "volatility": policy.vol_coefficient * np.maximum(0.0, vol - policy.reference_vol),
        "liquidity": policy.liquidity_addon * ones,
        "location": policy.location_addon * ones,
        "quality": policy.quality_addon * ones,
        "age": policy.age_addon_per_30d * max(0.0, age_days - policy.age_free_days) / 30.0 * ones,
        "concentration": np.where(np.asarray(inventory_share) > policy.concentration_threshold + 1e-12,
                                  policy.concentration_addon, 0.0) * ones,
        "unhedged": policy.unhedged_penalty * (1.0 - min(1.0, max(0.0, hedge_ratio))) * ones,
        "lender_stress": np.asarray(lender_stress, dtype=float) * ones,
    }


def dynamic_haircut(policy: HaircutPolicy, components: dict[str, np.ndarray]) -> np.ndarray:
    raw = sum(components[c] for c in HAIRCUT_COMPONENTS)
    return np.clip(raw, policy.min_haircut, policy.max_haircut)


def limit_haircut_increase(new: np.ndarray, previous: np.ndarray, max_increase: float | None) -> np.ndarray:
    """Contractual constraint on how far the haircut may rise at one redetermination."""
    if max_increase is None:
        return new
    return np.minimum(new, previous + max_increase)


def convention_warning(facility: FacilityConfig, policies: Sequence[HaircutPolicy]) -> str | None:
    if facility.convention == "haircut_and_advance" and facility.inventory_advance_rate < 1.0:
        worst = max((p.base for p in policies), default=0.0)
        if worst >= 0.10 and facility.inventory_advance_rate <= 0.90:
            return (f"Both a valuation haircut (base up to {worst:.0%}) and an inventory advance rate of "
                    f"{facility.inventory_advance_rate:.0%} are applied. Confirm the agreement states both; "
                    "otherwise this deducts the same price risk twice.")
    if facility.convention == "haircut_only" and facility.inventory_advance_rate < 1.0:
        return "The inventory advance rate is ignored under the haircut-only convention."
    return None


def cap_solve_matrix(bb_inv: np.ndarray, pre_caps: np.ndarray, cap_pct: float) -> tuple[np.ndarray, np.ndarray]:
    """Solve BB = BB_inv + sum_j min(a_j, cap x BB) per path when the pre-cap amounts differ by path.

    bb_inv (n,), pre_caps (n, c). Returns (BB, capped receivable contribution (n,)). Exact sorted-suffix closed
    form, the same algorithm as the legacy solver, vectorized across rows."""
    bb_inv = np.asarray(bb_inv, dtype=float)
    a = np.maximum(0.0, np.asarray(pre_caps, dtype=float))
    n, c = a.shape if a.ndim == 2 else (bb_inv.shape[0], 0)
    if c == 0 or cap_pct >= 1.0:
        ar = a.sum(axis=1) if c else np.zeros(n)
        return bb_inv + ar, ar
    a_sorted = -np.sort(-a, axis=1)
    suffix = np.concatenate([np.cumsum(a_sorted[:, ::-1], axis=1)[:, ::-1], np.zeros((n, 1))], axis=1)
    solved = np.zeros(n)
    found = np.zeros(n, dtype=bool)
    for k in range(c + 1):
        denom = 1.0 - k * cap_pct
        if denom <= 1e-12:
            break
        cand = (bb_inv + suffix[:, k]) / denom
        lim = cap_pct * cand
        upper_ok = np.ones(n, dtype=bool) if k == 0 else a_sorted[:, k - 1] >= lim - 1e-9
        lower_ok = np.ones(n, dtype=bool) if k == c else a_sorted[:, k] <= lim + 1e-9
        hit = upper_ok & lower_ok & ~found
        solved = np.where(hit, np.maximum(0.0, cand), solved)
        found |= hit
    ar = np.minimum(a, (cap_pct * solved)[:, None]).sum(axis=1)
    return bb_inv + ar, ar


def borrowing_base(
    facility: FacilityConfig,
    inventory_adjusted_value: np.ndarray,
    ar_pre_cap_by_counterparty: np.ndarray,
) -> dict[str, np.ndarray]:
    """Borrowing base per path from the adjusted inventory value (after haircuts) and the pre-cap eligible
    receivable amount per counterparty. Order: inventory advance rate, concentration cap (fixed point on the
    pre-reserve base), sublimits, reserves."""
    inv = np.asarray(inventory_adjusted_value, dtype=float)
    if facility.convention == "haircut_and_advance":
        inv = inv * facility.inventory_advance_rate
    if facility.inventory_sublimit is not None:
        inv = np.minimum(inv, facility.inventory_sublimit)
    _, ar = cap_solve_matrix(inv, ar_pre_cap_by_counterparty, facility.concentration_cap_pct)
    ar_pre = np.maximum(0.0, ar_pre_cap_by_counterparty).sum(axis=1) if np.ndim(ar_pre_cap_by_counterparty) == 2 \
        else np.zeros_like(inv)
    if facility.receivable_sublimit is not None:
        ar = np.minimum(ar, facility.receivable_sublimit)
    bb = np.maximum(0.0, inv + ar - facility.fixed_reserves)
    return {"inventory_contribution": inv, "receivable_contribution": ar, "concentration_cutback": ar_pre - ar,
            "borrowing_base": bb}


def availability(facility: FacilityConfig, reported_bb: np.ndarray, drawn: np.ndarray,
                 pending: np.ndarray | float = 0.0) -> dict[str, np.ndarray]:
    """Permitted utilization, undrawn availability (never negative) and deficiency (never negative)."""
    permitted = np.minimum(reported_bb, facility.commitment)
    undrawn = np.maximum(0.0, permitted - drawn - pending)
    deficiency = np.maximum(0.0, drawn - permitted)
    return {"permitted": permitted, "undrawn": undrawn, "deficiency": deficiency}
