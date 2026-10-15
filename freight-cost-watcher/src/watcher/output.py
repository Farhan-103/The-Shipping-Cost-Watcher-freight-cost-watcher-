"""Output module: write output.csv and diagnostics_all_weeks.csv."""
import csv
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# The exact header from sample_output_format_v2.csv
OUTPUT_HEADER = [
    'route', 'week_of', 'cost_per_tonne_km',
    'vs_own_history', 'vs_similar_routes',
    'flagged', 'matched_note_id', 'reason',
]

DIAGNOSTICS_HEADER = [
    'route', 'week_of', 'cost_per_tonne_km', 'route_type',
    'vs_own_history', 'vs_similar_routes',
    'history_weeks_used', 'peer_count',
    'own_pct', 'peer_pct',
    'is_candidate', 'retrieved_note_ids',
    'flagged', 'matched_note_id', 'reason',
]


def format_own_history(pct: float) -> str:
    """Format the vs_own_history column: '+35.5% vs this route's past average'."""
    sign = '+' if pct >= 0 else ''
    return f"{sign}{pct:.1f}% vs this route's past average"


def format_peer(pct: float) -> str:
    """Format the vs_similar_routes column: '+21.0% vs similar-length routes this week'."""
    sign = '+' if pct >= 0 else ''
    return f'{sign}{pct:.1f}% vs similar-length routes this week'


def write_output_csv(
    rows: list[dict],
    output_path: Path,
    all_rows: bool = False,
) -> None:
    """Write the graded output CSV.

    Uses Python's csv module with UTF-8 encoding and minimal quoting.
    By default, includes only candidate rows (flagged or justified).
    With all_rows=True, includes all route-weeks.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_HEADER,
            extrasaction='ignore',
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()

        for row in rows:
            out = {
                'route': row['route'],
                'week_of': row['week_of'],
                'cost_per_tonne_km': f"{row['cost_per_tonne_km']:.2f}",
                'vs_own_history': row['vs_own_history'],
                'vs_similar_routes': row['vs_similar_routes'],
                'flagged': row['flagged'],
                'matched_note_id': row.get('matched_note_id', ''),
                'reason': row.get('reason', ''),
            }
            writer.writerow(out)

    logger.info('Wrote %d rows to %s', len(rows), output_path)


def write_diagnostics_csv(
    rows: list[dict],
    output_path: Path,
) -> None:
    """Write diagnostics_all_weeks.csv with all 728 route-weeks."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=DIAGNOSTICS_HEADER,
            extrasaction='ignore',
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()

        for row in rows:
            out = {
                'route': row['route'],
                'week_of': row['week_of'],
                'cost_per_tonne_km': f"{row['cost_per_tonne_km']:.2f}",
                'route_type': row.get('route_type', ''),
                'vs_own_history': row.get('vs_own_history', ''),
                'vs_similar_routes': row.get('vs_similar_routes', ''),
                'history_weeks_used': row.get('history_weeks_used', ''),
                'peer_count': row.get('peer_count', ''),
                'own_pct': f"{row['own_pct']:.2f}" if pd.notna(row.get('own_pct')) else '',
                'peer_pct': f"{row['peer_pct']:.2f}" if pd.notna(row.get('peer_pct')) else '',
                'is_candidate': row.get('is_candidate', ''),
                'retrieved_note_ids': row.get('retrieved_note_ids', ''),
                'flagged': row.get('flagged', ''),
                'matched_note_id': row.get('matched_note_id', ''),
                'reason': row.get('reason', ''),
            }
            writer.writerow(out)

    logger.info('Wrote %d rows to %s', len(rows), output_path)
