"""Reason text generation: template-based and optional LLM rewriting."""
import logging
import re
from datetime import date, timedelta

from .notes import ParsedNote

logger = logging.getLogger(__name__)


def template_reason(
    flagged: str,
    matched_note: ParsedNote | None,
    route: str,
    week_monday: date,
    own_pct_str: str,
    candidates: list[ParsedNote],
    rejected_nearby: ParsedNote | None = None,
) -> str:
    """Build a template reason from facts only (no LLM).

    Three templates:
    1. Justified: cites the matching note.
    2. No candidate: no matching notes found.
    3. Candidate rejected: closest note doesn't explain the cost rise.
    """
    if flagged == 'No (justified)' and matched_note is not None:
        return (
            f'Matches note {matched_note.note_id} dated '
            f'{matched_note.start_date.isoformat()}: '
            f'{_summarize_note(matched_note.text)}. '
            f'The cost rise has a clear explanation.'
        )

    # For Yes rows, check if there's a nearby rejected note to mention
    if rejected_nearby is not None:
        return (
            f'The closest note ({rejected_nearby.note_id}, '
            f'{rejected_nearby.start_date.isoformat()}) '
            f'{_describe_rejection(rejected_nearby.text)} '
            f'-- it does not describe a reason for a cost rise on this route. '
            f'No genuine justification found; flagged for review.'
        )

    if candidates:
        # There were candidates but none were valid
        first = candidates[0]
        return (
            f'The closest note ({first.note_id}, '
            f'{first.start_date.isoformat()}) '
            f'{_describe_rejection(first.text)} '
            f'-- it does not describe a reason for a cost rise on this route. '
            f'No genuine justification found; flagged for review.'
        )

    return (
        'No matching note found for this route or date range. '
        'Cost rise looks unexplained and worth a human review.'
    )


def _summarize_note(text: str) -> str:
    """Create a short summary of the note text for the reason."""
    # Take the key phrase from the note
    text = text.strip()
    # Truncate to a reasonable length
    if len(text) > 150:
        # Try to cut at a sentence boundary
        cut = text[:150].rfind('.')
        if cut > 50:
            return text[:cut + 1].strip()
        return text[:150].strip() + '...'
    return text


def _describe_rejection(text: str) -> str:
    """Describe what a rejected note says, for context."""
    text = text.strip().lower()
    if 'stable' in text or 'normal' in text:
        return 'mentions stable demand with no major disruptions'
    if 'improved' in text or 'completed' in text:
        return 'mentions improved or completed conditions'
    if 'not significantly' in text or 'not affected' in text:
        return 'says costs were not significantly affected'
    if 'no significant' in text or 'no disruption' in text:
        return 'reports no significant disruptions'
    if 'returned to normal' in text:
        return 'says conditions returned to normal'
    if 'no rate change' in text or 'absorbed' in text:
        return 'says costs were absorbed without a rate change'
    return 'does not describe a reason for a cost rise'


def find_nearby_rejected_note(
    all_notes: list[ParsedNote],
    route: str,
    week_monday: date,
    matched_note_id: str | None,
    is_valid_fn,
    days_range: int = 14,
) -> ParsedNote | None:
    """Find a route-matching or All Routes note within `days_range` days.

    Only returns notes that were NOT accepted (rejected).
    Used for the optional 'considered and rejected' mention in Yes reasons.
    """
    week_sunday = week_monday + timedelta(days=6)

    nearby = []
    for note in all_notes:
        # Skip if it's the matched note
        if matched_note_id and note.note_id == matched_note_id:
            continue

        # Must apply to this route or All Routes
        if not note.matches_route(route):
            continue

        # Start date within days_range of the week
        days_diff = abs((note.start_date - week_monday).days)
        if days_diff > days_range:
            # Also check if start is near the end of the week
            days_diff_end = abs((note.start_date - week_sunday).days)
            if days_diff_end > days_range:
                continue

        # Must be rejected
        if is_valid_fn(note.text):
            continue

        nearby.append((days_diff, note))

    if not nearby:
        return None

    nearby.sort(key=lambda x: (x[0], x[1].note_id))
    return nearby[0][1]


def validate_llm_reason(
    llm_text: str,
    own_pct_str: str,
    flagged: str,
    matched_note_id: str | None,
    matched_note_start: date | None,
    all_note_ids: list[str],
) -> bool:
    """Validate an LLM-rewritten reason text.

    Rules:
    - Single line (no newlines)
    - Contains the own-history percentage string
    - For justified rows: contains matched note id and start date
    - Contains no other note id
    """
    if '\n' in llm_text:
        return False

    # Must contain the own-history percentage
    if own_pct_str not in llm_text:
        return False

    if flagged == 'No (justified)' and matched_note_id:
        if matched_note_id not in llm_text:
            return False
        if matched_note_start and matched_note_start.isoformat() not in llm_text:
            return False

    # Check no other note id appears
    for nid in all_note_ids:
        if nid == matched_note_id:
            continue
        if nid in llm_text:
            return False

    return True
