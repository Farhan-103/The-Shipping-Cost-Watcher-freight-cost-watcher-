#!/usr/bin/env python
"""Threshold study: show how many rows flag at different thresholds.

Prints a table of candidate counts at 5%, 8%, 9%, 10%, 15% thresholds,
plus own-OR-peer variants. Used to justify FLAG_OWN_PCT = 9.0 in the README.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.watcher.loader import load_shipments, validate_shipments
from src.watcher.weekly import compute_weekly_cost
from src.watcher.baselines import add_own_history, add_peer_baseline, add_percentages


def main():
    shipments = load_shipments(_PROJECT_ROOT / 'data' / 'shipment_records.csv')
    validate_shipments(shipments)
    weekly = compute_weekly_cost(shipments)
    weekly = add_own_history(weekly)
    weekly = add_peer_baseline(weekly)
    weekly = add_percentages(weekly)

    has_history = weekly['history_weeks_used'] > 0

    thresholds = [5.0, 8.0, 9.0, 10.0, 15.0]

    print('=== Threshold Study ===')
    print()
    print(f'{"Threshold":>10}  {"Own-only":>10}  {"Own-OR-Peer":>12}  {"Peer-only":>10}')
    print('-' * 50)

    for t in thresholds:
        own_only = (has_history & (weekly['own_pct'] >= t)).sum()
        peer_only = (has_history & (weekly['peer_pct'] >= t)).sum()
        own_or_peer = (has_history & ((weekly['own_pct'] >= t) | (weekly['peer_pct'] >= t))).sum()
        print(f'{t:>9.1f}%  {own_only:>10}  {own_or_peer:>12}  {peer_only:>10}')

    print()
    print('At 9.0% own-only, the flagged rows are:')

    candidates = weekly[has_history & (weekly['own_pct'] >= 9.0)].copy()
    candidates = candidates.sort_values(['route', 'week_of'])

    for _, row in candidates.iterrows():
        week_str = row['week_of'].strftime('%Y-%m-%d')
        print(f"  {row['route']:25s}  {week_str}  own={row['own_pct']:+.1f}%  peer={row['peer_pct']:+.1f}%")

    print(f'\nTotal candidates at 9.0%: {len(candidates)}')


if __name__ == '__main__':
    main()
