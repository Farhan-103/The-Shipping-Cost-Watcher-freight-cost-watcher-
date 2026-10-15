# Freight Cost Watcher — Evaluation Report

## Executive Summary

- **Decision Accuracy**: **100.0%** (9/9 labeled cases)
- **Wrongly Justified Rows (Hard Gate = 0)**: **0**
- **Retrieval Top-3 Hit Rate**: **100.0%** (3/3 justified cases)
- **Real Notes Classification**: **100.0%** (10/10)
- **Tricky Notes Classification**: **100.0%** (10/10)
- **Heldout Notes Classification**: **100.0%** (6/6)

---

## 1. Candidate Decision Evaluation (Labeled Cases)

| Route | Week | Expected Flagged | Actual Flagged | Expected Note | Actual Note | Match | Top 3 Retrieved |
|---|---|---|---|---|---|:---:|---|
| Delhi-Jaipur | 2024-11-11 | Yes | Yes | - | - | PASS | none |
| Delhi-Jaipur | 2024-11-18 | Yes | Yes | - | - | PASS | none |
| Ahmedabad-Mumbai | 2025-01-20 | No (justified) | No (justified) | N002 | N002 | PASS | N002 |
| Chennai-Bangalore | 2025-02-24 | No (justified) | No (justified) | N001 | N001 | PASS | N001 |
| Chennai-Bangalore | 2025-03-03 | No (justified) | No (justified) | N001 | N001 | PASS | N001 |
| Chennai-Bangalore | 2025-03-10 | Yes | Yes | - | - | PASS | none |
| Chennai-Bangalore | 2025-03-17 | Yes | Yes | - | - | PASS | N009 |
| Mumbai-Delhi | 2025-05-12 | Yes | Yes | - | - | PASS | none |
| Mumbai-Pune | 2025-09-15 | Yes | Yes | - | - | PASS | none |

---

## 2. Classifier Performance on Fixture Datasets

### Real Notes (10 notes)
- Score: **10/10 (100.0%)**
- Target: 10/10 (N001-N003 Accept, N004-N010 Reject)

### Tricky Notes Fixture (10 notes)
- Score: **10/10 (100.0%)**

| Text | Expected | Keyword Pred | Result |
|---|:---:|:---:|:---:|
| Costs may rise next month. | Reject | Reject | PASS |
| There was no increase in rates. | Reject | Reject | PASS |
| Costs rose sharply last year. | Reject | Reject | PASS |
| Costs were higher. | Reject | Reject | PASS |
| The surcharge was waived this week. | Reject | Reject | PASS |
| Prices stayed flat despite the diesel increase. | Reject | Reject | PASS |
| A fuel surcharge raised rates 8% on this route. | Accept | Accept | PASS |
| Heavy rain forced detours pushing up trip costs. | Accept | Accept | PASS |
| Transporters charged more due to a strike. | Accept | Accept | PASS |
| Costs went up because of a bridge closure. | Accept | Accept | PASS |

### Fresh Heldout Notes (6 notes)
- Score: **6/6 (100.0%)**

| Text | Expected | Keyword Pred | Result |
|---|:---:|:---:|:---:|
| Freight rates jumped 12% after a landslide blocked the highway. | Accept | Accept | PASS |
| The toll hike added a surcharge to all northbound shipments. | Accept | Accept | PASS |
| A port congestion levy pushed up container handling costs. | Accept | Accept | PASS |
| Diesel subsidy removal raised base transportation rates by 4%. | Accept | Accept | PASS |
| Severe fog led to delays and overtime charges increasing costs. | Accept | Accept | PASS |
| Weather conditions improved and rates returned to pre-storm levels. | Reject | Reject | PASS |
