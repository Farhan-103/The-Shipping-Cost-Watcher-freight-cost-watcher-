#!/usr/bin/env python
"""Audit justified rows: verify every 'No (justified)' row is genuine.

For each justified row, checks:
1. The matched note exists.
2. The note applies to this route (or All Routes).
3. The note's window overlaps the week.
4. The classifier accepts the note.
5. The reason text names that note id and no other.

Exits non-zero on any failure.
"""
import csv
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.watcher.loader import load_notes
from src.watcher.notes import parse_notes
from src.watcher.classifier import keyword_says_cost_up
from datetime import date


def main():
    output_path = _PROJECT_ROOT / 'output' / 'output.csv'
    notes_path = _PROJECT_ROOT / 'data' / 'context_notes.csv'

    if not output_path.exists():
        print(f'ERROR: {output_path} not found. Run the pipeline first.')
        sys.exit(1)

    # Load notes
    import pandas as pd
    notes_df = pd.read_csv(notes_path, parse_dates=['date'])
    parsed_notes = parse_notes(notes_df)
    note_map = {n.note_id: n for n in parsed_notes}
    all_note_ids = set(note_map.keys())

    # Read output
    with open(output_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    justified = [r for r in rows if r['flagged'] == 'No (justified)']
    print(f'Found {len(justified)} justified rows to audit.')

    failures = []

    for row in justified:
        route = row['route']
        week_of = row['week_of']
        note_id = row['matched_note_id']
        reason = row['reason']
        week_monday = date.fromisoformat(week_of)

        checks = []

        # 1. Note exists
        if note_id not in note_map:
            checks.append(f'Note {note_id} does not exist')
        else:
            note = note_map[note_id]

            # 2. Route match
            if not note.matches_route(route):
                checks.append(
                    f'Note {note_id} applies to {note.applies_to}, '
                    f'not {route}'
                )

            # 3. Window overlap
            if not note.overlaps_week(week_monday):
                checks.append(
                    f'Note {note_id} window ({note.start_date} to '
                    f'{note.end_date}) does not overlap week {week_of}'
                )

            # 4. Classifier accepts
            if not keyword_says_cost_up(note.text):
                checks.append(
                    f'Classifier rejects note {note_id}'
                )

        # 5. Reason names the note id
        if note_id and note_id not in reason:
            checks.append(f'Reason does not mention {note_id}')

        # 5b. Reason names no other note id
        for other_id in all_note_ids:
            if other_id != note_id and other_id in reason:
                checks.append(f'Reason mentions unmatched note {other_id}')

        if checks:
            failures.append((route, week_of, note_id, checks))
            for check in checks:
                print(f'  FAIL: {route}/{week_of}: {check}')
        else:
            print(f'  OK:   {route}/{week_of} -> {note_id}')

    print()
    if failures:
        print(f'AUDIT FAILED: {len(failures)} justified rows have issues.')
        sys.exit(1)
    else:
        print(f'AUDIT PASSED: All {len(justified)} justified rows verified.')
        sys.exit(0)


if __name__ == '__main__':
    main()
