"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations

# Industry/cloud-observed eviction rates per hour (Extension 1)
# H100/H200 clusters have high demand & dedicated capacity pools (~3-4% eviction).
# Consumer/edge-class GPUs (A10G) experience higher volatility & host evictions (~12%).
GPU_INTERRUPT_RATES: dict[str, float] = {
    "H100": 0.03,
    "H200": 0.04,
    "A100": 0.05,
    "L4": 0.08,
    "A10G": 0.12,
    "B200": 0.04,
    "MI300X": 0.06,
}


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def recommend_tier(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = 0.45,
    gpu_type: str | None = None,
    job_days: int | None = None,
    max_acceptable_interrupt_rate: float = 0.10,
    min_reserved_days: int = 14,
) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    Enhanced Policy (Extension 1):
      1. If interruptible:
         - If gpu_type has high eviction risk (> max_acceptable_interrupt_rate, e.g., A10G at 12%),
           and duty cycle is low, spot thrashing / checkpoint overhead may degrade SLA.
           However, for fault-tolerant dev/batch workloads under 24h, spot remains dominant.
         - For interruptible & hours_per_day < 24 -> 'spot'.
      2. If steady (duty >= break_even):
         - If job duration is very short (job_days < min_reserved_days), locking into reserved
           carries commitment penalty, so fall back to 'on_demand'.
         - Otherwise -> 'reserved'.
      3. Otherwise -> 'on_demand'.
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)

    if interruptible and hours_per_day < 24:
        return "spot"

    if duty >= be:
        if job_days is not None and job_days < min_reserved_days:
            return "on_demand"
        return "reserved"

    return "on_demand"


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings = on_demand_cost - spot_cost
    savings_pct = (savings / on_demand_cost * 100.0) if on_demand_cost else 0.0
    return {
        "job_hours": job_hours,
        "effective_hours": round(effective_hours, 1),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }


# =========================================================================
# Extension 3: Prompt Caching Economics (cache_is_worth_it)
# =========================================================================

def cache_break_even_reads(
    write_cost_per_m: float,
    base_input_price_per_m: float = 3.00,
    read_discount: float = 0.10,
) -> float:
    """Calculate minimum number of cache reads required to break even against write cost.

    Each cached read saves: base_input_price_per_m * (1.0 - read_discount).
    Break-even reads = write_cost_per_m / savings_per_read.
    """
    savings_per_read = base_input_price_per_m * (1.0 - read_discount)
    if savings_per_read <= 0:
        return float("inf")
    return write_cost_per_m / savings_per_read


def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost_per_m: float,
    base_input_price_per_m: float = 3.00,
    read_discount: float = 0.10,
) -> bool:
    """Prompt caching is economically worth it ONLY when read savings exceed initial write cost.

    In production (e.g. Anthropic, Gemini, OpenAI):
      - Cache write carries an initial premium (or storage charge per hour).
      - If a prefix is read fewer times than the break-even threshold, caching loses money.
    """
    be_reads = cache_break_even_reads(write_cost_per_m, base_input_price_per_m, read_discount)
    return avg_cache_reads > be_reads
