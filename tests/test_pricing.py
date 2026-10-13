import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing


def test_discount_stack_is_multiplicative():
    assert abs(pricing.discount_stack(batch=True, cache_hit_frac=1.0) - 0.05) < 1e-9   # ~95% off
    assert pricing.discount_stack() == 1.0
    assert abs(pricing.discount_stack(batch=True) - 0.5) < 1e-9


def test_break_even():
    assert abs(pricing.break_even_utilization(0.45) - 0.55) < 1e-9
    assert pricing.break_even_utilization(0.0) == 1.0


def test_recommend_tier():
    assert pricing.recommend_tier(2, True) == "spot"
    assert pricing.recommend_tier(24, False) == "reserved"
    assert pricing.recommend_tier(4, False) == "on_demand"


def test_request_cost_and_cache():
    full = pricing.request_cost(1000, 1000, 3.0, 15.0)
    cached = pricing.request_cost(1000, 1000, 3.0, 15.0, cached_in=1000)
    assert cached < full                       # caching reduces cost
    batched = pricing.request_cost(1000, 1000, 3.0, 15.0, batch=True)
    assert abs(batched - full * 0.5) < 1e-9    # batch = -50%


def test_spot_checkpoint_saves():
    res = pricing.spot_checkpoint_cost(100, 1.5, 2.5)
    assert res["spot_cost"] < res["on_demand_cost"]
    assert res["savings_pct"] > 0


def test_extension1_recommend_tier_with_horizon_and_gpu():
    # Duty >= break-even but job_days < 14 (transient task) -> avoids long commit lock-in -> on_demand
    assert pricing.recommend_tier(24, False, job_days=5, min_reserved_days=14) == "on_demand"
    # Duty >= break-even and job_days >= 14 -> reserved
    assert pricing.recommend_tier(24, False, job_days=30, min_reserved_days=14) == "reserved"
    # Spot with low interruption risk H100
    assert pricing.recommend_tier(20, True, gpu_type="H100") == "spot"


def test_extension3_cache_economics():
    # Break-even reads calculation:
    # If base_input = $3.00, 90% discount -> saving per read = $2.70.
    # If write_cost = $3.75 (1.25x), break-even = 3.75 / 2.70 = 1.3888...
    be = pricing.cache_break_even_reads(write_cost_per_m=3.75, base_input_price_per_m=3.00, read_discount=0.10)
    assert abs(be - (3.75 / 2.70)) < 1e-6

    # If avg reads = 1 (< 1.39) -> cache loses money (False)
    assert not pricing.cache_is_worth_it(avg_cache_reads=1.0, write_cost_per_m=3.75, base_input_price_per_m=3.00)

    # If avg reads = 5 (> 1.39) -> cache is profitable (True)
    assert pricing.cache_is_worth_it(avg_cache_reads=5.0, write_cost_per_m=3.75, base_input_price_per_m=3.00)
