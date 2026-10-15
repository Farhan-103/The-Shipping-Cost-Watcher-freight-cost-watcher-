# Freight Cost Watcher — 10-Minute Walkthrough Outline

This document outlines an interactive 10-minute presentation of the Freight Cost Watcher system for leadership, engineering peers, and logistics operations stakeholders.

---

## 1. Context & Business Problem (1.5 mins)

- **The Problem**: Freight procurement teams face sudden rate spikes across hundreds of lane-weeks. Manual review of thousands of shipments is bottlenecked, reactive, and vulnerable to missed cost anomalies.
- **The Solution**: An automated, deterministic intelligence pipeline that ingests trip-level freight records, computes rigorous lane baselines, flags statistically anomalous cost surges, cross-references internal operational context notes, and generates trustworthy, explainable verdicts.
- **Guiding Principle**: *"Trustworthy over confident."* A confident wrong reason is catastrophic in automated freight auditing. An LLM must **never** decide numbers, thresholds, verdict flags, or matched IDs. All financial figures and candidate selections are strictly deterministic.

---

## 2. Architecture & Data Flow (2.5 mins)

```
shipment_records.csv ──► Weekly Grouping & Aggregation ──► Baseline Calculations
                                                                  │
                                                ┌─────────────────┘
                                                ▼
                                         Flagging Engine (>= 9.0%)
                                                │
                                                ▼ (Candidates Only)
context_notes.csv ──► Strict Window & ──► RAG Retriever (ChromaDB + Python filter)
                      Applicability             │
                                                ▼
                                         Dual Classifier
                                  (Keyword Guards + Temperature-0 LLM)
                                                │
                                                ▼
                                      Deterministic Arbiter
                                      (Flagged: Yes / No (justified))
                                                │
                                                ▼
                                    Template / Grounded Reasons
                                                │
                                                ▼
                                     output/output.csv &
                                 diagnostics_all_weeks.csv
```

1. **Deterministic Core**:
   - `route = origin + '-' + destination`
   - `week_of = Monday of week (ISO 8601)`
   - Full-precision formula: `sum(freight_cost_inr) / sum(quantity_tonnes * distance_km)` (sum first, divide once).
   - Trailing 8-week history baseline (strictly past, no look-ahead; weeks 2–8 use available history; week 1 uncompared).
   - Peer comparison: average of concurrent routes with the same `route_type` (excluding route itself).

2. **Flagging Mechanism**:
   - Candidate threshold: `own_pct >= 9.0%`.
   - Discovered through empirical threshold sweep (5% produces 50 noisy rows; 15% misses Mumbai-Pune anomaly).
   - Peer filtering kept optional (Mumbai-Pune has structural premia over Delhi-Jaipur).

---

## 3. Retrieval & Grounded Verification (2 mins)

- **Strict Date & Route Scoping**:
   - Context notes contain explicit windows (e.g. N001: Feb 24 – Mar 8) or default 1-week windows (`start` to `start + 6 days`).
   - N003 Diesel increase: intentionally bounded to 1 week (May 5–11) to prevent runaway false justifications across the entire multi-year dataset.
   - Dual-layer retrieval: Vector similarity search (ChromaDB `all-MiniLM-L6-v2`) cross-verified against deterministic Python date/route logic.
- **Two-Gate Classifier**:
   - Gate 1: Deterministic keyword engine with negative and hedge guardrails (`not`, `waived`, `flat`, `despite`).
   - Gate 2: Temperature-0 LLM schema enforcement (validates explicit cost increase assertion, named cause, and absence of speculation).
   - Both gates must agree; disagreements default safely to `Yes` (unjustified, flagged for human review).

---

## 4. Verification & Testing Evidence (2 mins)

- **Deterministic Snapshot Reproduction**:
   - Exact numerical reproduction of the 3 required sample rows:
     - `Delhi-Jaipur` (2024-11-11): 4.17 INR/t-km, +35.5% vs own, +21.0% vs peers (`Yes`)
     - `Ahmedabad-Mumbai` (2025-01-20): 3.29 INR/t-km, +29.5% vs own, +22.5% vs peers (`No (justified)`, N002)
     - `Mumbai-Pune` (2025-09-15): 3.98 INR/t-km, +9.2% vs own, +23.6% vs peers (`Yes`)
   - 9-row golden regression snapshot matches with 100% precision.
- **Hard Safety Gate**:
   - **Wrongly justified rows: 0**. Zero fabricated note citations.
   - Chennai-Bangalore (2025-03-10) correctly stays `Yes` because the flood window expired March 8.
- **Reproducibility**:
   - `scripts/repro_check.py`: 3 consecutive runs produce byte-for-byte identical output on all graded columns.
   - Offline pytest suite (100% test coverage of formulas, date windows, classifier guards, and edge cases).

---

## 5. Token Efficiency & Operating Economics (1 min)

- **Zero-Waste LLM Integration**:
   - Disk-cached responses keyed by SHA256 hashes of provider, model, prompt version, and content.
   - Cold run: exactly 10 classification calls (~1,400 input tokens, ~300 output tokens; total cost < $0.0003 USD).
   - Warm runs: **0 LLM calls**, $0.00 cost.
   - Local vector embeddings run on CPU at $0.00 cost.
   - Hard circuit breaker: `MAX_LLM_CALLS = 40`.

---

## 6. Key Judgment Calls & Trade-offs (1 min)

1. **Candidate-Only Output Contract**: The primary `output.csv` delivers actionable exceptions (9 candidate rows), while `diagnostics_all_weeks.csv` retains full transparency across all 728 route-weeks.
2. **N003 1-Week Scope**: Treating "starting this week" as a bounded 7-day impact rather than perpetual blanket justification.
3. **No Effect-Size LLM Reasoning**: Left as a future extension rather than risking subjective LLM arithmetic.

---

## Q&A Discussion Points (Ready for Panel)

- *Why not allow the LLM to write custom SQL or query embeddings freely?*
  Because nondeterminism in financial reconciliation creates compliance risks; deterministic filtering guarantees auditable reproducibility.
- *How would this scale to 50,000 shipments/day?*
  Pandas aggregations transition directly to DuckDB / Polars with sub-second execution; ChromaDB scales linearly with metadata partition pruning.
