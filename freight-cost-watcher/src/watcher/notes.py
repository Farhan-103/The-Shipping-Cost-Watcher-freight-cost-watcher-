"""Parse context notes: extract date windows, route applicability, text."""
import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ParsedNote:
    """A context note with its parsed date window and route."""
    note_id: str
    start_date: date
    end_date: date
    applies_to: str   # route name or 'All Routes'
    text: str

    def overlaps_week(self, week_monday: date) -> bool:
        """Check if this note's window overlaps with the given week (Mon-Sun)."""
        week_sunday = week_monday + timedelta(days=6)
        return self.start_date <= week_sunday and self.end_date >= week_monday

    def matches_route(self, route: str) -> bool:
        """Check if this note applies to the given route."""
        return self.applies_to == 'All Routes' or self.applies_to == route


def _parse_end_date(note_id: str, start_date: date, text: str) -> date:
    """Extract end date from note text, or default to start + 6 days.

    Rules:
    - If the text gives an explicit range like 'from Feb 24 to Mar 8',
      the end is that date (year from start_date; if end < start, add a year).
    - Otherwise the window is one week: start to start + 6 days.
    - N003 says 'starting this week'; treat as one week only.
      (Judgment call: an open-ended window would wrongly justify routes
       after May 2025, including the Mumbai-Pune sample row which must stay Yes.)
    """
    t = text.lower()

    # Look for explicit date ranges like "from Feb 24 to Mar 8"
    # or "Feb 24 to Mar 8" patterns
    range_patterns = [
        # "from Mon DD to Mon DD"
        r'from\s+(\w+)\s+(\d{1,2})\s+to\s+(\w+)\s+(\d{1,2})',
        # "Mon DD to Mon DD"
        r'(\w+)\s+(\d{1,2})\s+to\s+(\w+)\s+(\d{1,2})',
    ]

    month_map = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
        'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
        'may': 5, 'jun': 6, 'june': 6,
        'jul': 7, 'july': 7, 'aug': 8, 'august': 8,
        'sep': 9, 'september': 9, 'oct': 10, 'october': 10,
        'nov': 11, 'november': 11, 'dec': 12, 'december': 12,
    }

    for pattern in range_patterns:
        match = re.search(pattern, t)
        if match:
            end_month_str = match.group(3).lower()
            end_day = int(match.group(4))

            if end_month_str in month_map:
                end_month = month_map[end_month_str]
                year = start_date.year

                try:
                    end_date = date(year, end_month, end_day)
                except ValueError:
                    # Invalid date, fall through to default
                    continue

                # If end < start, add a year
                if end_date < start_date:
                    end_date = date(year + 1, end_month, end_day)

                logger.debug(
                    'Note %s: explicit end date %s from text',
                    note_id, end_date,
                )
                return end_date

    # Default: one week (start + 6 days)
    default_end = start_date + timedelta(days=6)
    logger.debug('Note %s: default end date %s (start + 6)', note_id, default_end)
    return default_end


def parse_notes(df: pd.DataFrame) -> list[ParsedNote]:
    """Parse all notes from the DataFrame into ParsedNote objects.

    Returns a list sorted by note_id.
    """
    notes = []
    for _, row in df.iterrows():
        note_id = row['note_id']
        start_date = row['date'].date() if hasattr(row['date'], 'date') else row['date']
        applies_to = row['applies_to']
        text = row['note']

        end_date = _parse_end_date(note_id, start_date, text)

        notes.append(ParsedNote(
            note_id=note_id,
            start_date=start_date,
            end_date=end_date,
            applies_to=applies_to,
            text=text,
        ))

    notes.sort(key=lambda n: n.note_id)

    logger.info('Parsed %d notes', len(notes))
    for n in notes:
        logger.debug(
            '  %s: %s to %s, applies_to=%s',
            n.note_id, n.start_date, n.end_date, n.applies_to,
        )

    return notes
