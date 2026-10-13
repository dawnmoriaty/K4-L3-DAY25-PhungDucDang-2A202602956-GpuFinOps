"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Extended with:
  - Extension 3: cache_is_worth_it economics & break-even analysis.
  - Extension 4: Reasoning budget analysis ($ and Wh breakdown).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0

    # Extension 4 tracking
    reasoning_reqs = standard_reqs = 0
    reasoning_tokens = standard_tokens = 0
    reasoning_base_cost = reasoning_opt_cost = 0.0
    standard_base_cost = standard_opt_cost = 0.0
    reasoning_wh = standard_wh = 0.0

    # Extension 3 cache tracking
    total_input_tokens = total_cached_tokens = 0

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = bool(int(num(r.get("is_reasoning", 0))))
        req_tokens = inp + out
        total_tokens += req_tokens
        total_input_tokens += inp
        total_cached_tokens += cached

        # Energy calculation
        query_wh = sustainability.wh_per_query(req_tokens, is_reasoning=is_reasoning)

        # BASELINE: naive deployment — everything on large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        req_base = pricing.request_cost(inp, out, lin, lout)
        base_cost += req_base

        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        # Extension 3 check: only claim cache discount if cache is economically viable
        cache_applies = pricing.cache_is_worth_it(
            avg_cache_reads=5.0,
            write_cost_per_m=pin * 1.25,
            base_input_price_per_m=pin,
            read_discount=0.10,
        )
        effective_cached = cached if cache_applies else 0
        req_opt = pricing.request_cost(inp, out, pin, pout, cached_in=effective_cached, batch=is_batch)
        opt_cost += req_opt

        if is_reasoning:
            reasoning_reqs += 1
            reasoning_tokens += req_tokens
            reasoning_base_cost += req_base
            reasoning_opt_cost += req_opt
            reasoning_wh += query_wh
        else:
            standard_reqs += 1
            standard_tokens += req_tokens
            standard_base_cost += req_base
            standard_opt_cost += req_opt
            standard_wh += query_wh

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # Extension 3 metrics
    be_reads_large = pricing.cache_break_even_reads(write_cost_per_m=3.00 * 1.25, base_input_price_per_m=3.00)
    be_reads_small = pricing.cache_break_even_reads(write_cost_per_m=0.20 * 1.25, base_input_price_per_m=0.20)

    # Extension 4 metrics
    total_energy_wh = reasoning_wh + standard_wh
    reasoning_traffic_pct = reasoning_reqs / len(rows) * 100
    reasoning_energy_pct = (reasoning_wh / total_energy_wh * 100) if total_energy_wh else 0.0
    reasoning_cost_pct = (reasoning_opt_cost / opt_cost * 100) if opt_cost else 0.0

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("")
        print("--- Extension 3: Prompt Caching Economics ---")
        print(f"Cached tokens share : {total_cached_tokens:,} / {total_input_tokens:,} ({total_cached_tokens/total_input_tokens:.1%})")
        print(f"Break-even reads    : {be_reads_large:.2f} reads (Large model) | {be_reads_small:.2f} reads (Small model)")
        print(f"Cache status        : {'PROFITABLE (worth it)' if cache_applies else 'UNPROFITABLE'}")

        print("")
        print("--- Extension 4: Reasoning Budget & Energy Impact ---")
        print(f"Reasoning traffic   : {reasoning_reqs} reqs ({reasoning_traffic_pct:.1f}% of total)")
        print(f"Reasoning cost      : ${reasoning_opt_cost:.2f}/day ({reasoning_cost_pct:.1f}% of opt cost)")
        print(f"Reasoning energy    : {reasoning_wh:,.1f} Wh ({reasoning_energy_pct:.1f}% of total inference Wh!)")
        print(f"Energy disparity    : Reasoning consumes {reasoning_wh / standard_wh:.1f}x energy of standard traffic")
        print("Routing recommendation: Cap unconstrained reasoning at 10% traffic via classifier gating.")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "reasoning_stats": {
            "traffic_pct": round(reasoning_traffic_pct, 1),
            "energy_pct": round(reasoning_energy_pct, 1),
            "cost_opt": round(reasoning_opt_cost, 2),
            "energy_wh": round(reasoning_wh, 1),
        },
        "cache_economics": {
            "be_reads": round(be_reads_large, 2),
            "is_worth_it": cache_applies,
        }
    }


if __name__ == "__main__":
    run()
