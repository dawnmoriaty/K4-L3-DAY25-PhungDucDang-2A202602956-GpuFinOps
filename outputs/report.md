# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,626  
**Projected savings:** $12,507  (**46%**)

## Executive Summary & Unit Economics

| Metric | Baseline | Optimized | Improvement |
|---|---|---|---|
| Monthly Total Spend | $27,133 | $14,626 | -46.1% |
| Inference Unit Cost | $6.488 / 1M-token | $1.126 / 1M-token | -82.6% |
| Purchasing Efficiency | 100% On-Demand | Spot + 3-Yr Reserved | -39.1% |
| FinOps Maturity Gate | Showback | Chargeback Ready (92% tagged) | Unlocked |

## Savings by FinOps Lever

| Lever | Savings (USD) | Share of Savings | Primary Mechanism |
|---|---|---|---|
| Inference (cascade/cache/batch) | $1,212 | 9.7% | Cascade to small model (90% off), prompt cache (90% hit off), batch API (50% off) |
| Purchasing (spot/reserved) | $10,040 | 80.3% | Shift fault-tolerant training to Spot; 24/7 steady infer to 3-Yr Reserved (45% off) |
| Right-size util-lies | $655 | 5.2% | Downgrade memory-bound decode GPUs (H100 -> A100 / A10G -> L4) |
| Kill idle GPUs | $600 | 4.8% | Terminate orphaned/idle GPU instances (<10% utilization) |

## Deep-Dive: Root Cause Analysis of the 'GPU-Util Lie'

- **The Illusion:** `gpu-h100-4` reports **98.2% GPU-Util** via standard `nvidia-smi` telemetry.
- **The Reality (MFU = 0.194):** Less than 20% of theoretical H100 FP16/BF16 Tensor Core FLOPs are actively computing.
- **Underlying Hardware Mechanism:**
  - LLM auto-regressive token generation (decode) is fundamentally **memory-bandwidth bound** with arithmetic intensity ~1-2 FLOP/byte, far below H100's ridge point (~295 FLOP/byte).
  - The Streaming Multiprocessors (SMs) report active execution cycles while stalled waiting for high-bandwidth memory (HBM3) transfers or suffering kernel launch latencies.
- **Financial Impact:** Paying $2.50/hr for flagship H100 compute to run memory-starved tasks wastes over $655/month per node. Right-sizing memory-bound decode to cost-efficient accelerators (A100/L4) preserves identical throughput at a fraction of the cost.

## Extensions & Advanced FinOps Policies

### 1. Eviction-Aware Purchasing Tier Recommender (Extension 1)
- Incorporates empirical cloud eviction rates: H100 (3%), A100 (5%), L4 (8%), A10G (12%).
- Workloads with high eviction risk (>10%) and developer-interactive constraints (e.g. `job-dev-sandbox`) are governed to avoid developer downtime, while batch training leverages spot safely with automated checkpointing.

### 2. Prompt Caching Economics (Extension 3)
- Evaluates write storage cost vs read discount: Break-even threshold is **1.39 reads per prefix**.
- In NimbusAI workload, system prompts and RAG contexts have average reuse of >5 reads, confirming caching is highly profitable and lowers $/1M-token dramatically.

### 3. Reasoning Budget & Energy Impact (Extension 4)
- **Traffic share:** Reasoning queries account for **8.4% of requests** and **16.5% of tokens**.
- **Energy consumption:** Reasoning consumes **29,787.7 Wh (94.0% of total inference energy!)** due to the ~80x chain-of-thought compute multiplier.
- **Governance Policy:** Enforce task-complexity routing to cap reasoning at <=10% traffic, avoiding quadratic energy spikes.

## Sustainability & Green Grid Strategy

- **Energy per query (Standard):** 0.24 Wh
- **Carbon per query:** 0.091 gCO2e (at us-east-1 grid intensity)
- **Cheapest & Cleanest Recommended Region:** **europe-north1** (Hydro-powered Norway: 30 gCO2/kWh vs Poland 660 gCO2/kWh).
- **Strategic Recommendation:** Route delay-tolerant batch jobs and offline training to `europe-north1` to slash carbon emissions by **92%** while lowering electricity costs.

## Prioritized Action Plan for NimbusAI Leadership

1. **P0 (Immediate - Day 1): Kill Idle GPUs & Enable Automated Eviction Cleaners**
   - Monthly Savings: **$600** | Effort: Zero risk, instant termination of idle nodes.
2. **P1 (Week 1): Shift Fault-Tolerant Training to Spot + 3-Yr Commit Steady Inference**
   - Monthly Savings: **$10,040** (Largest savings lever: 80.3% of total savings) | Effort: Configure S3/GCS checkpoint sync.
3. **P2 (Week 2): Deploy Cascade Routing, Prompt Caching & Batch API Gate**
   - Monthly Savings: **$1,212** | Unit cost drop: from **$6.488** to **$1.126** per 1M-token.
4. **P3 (Month 1): Right-Size Memory-Bound Decode Workloads**
   - Monthly Savings: **$655** | Downgrade over-provisioned H100 decode nodes to A100/L4.

_Figures are June-2026 as-of snapshots; re-baseline before acting._