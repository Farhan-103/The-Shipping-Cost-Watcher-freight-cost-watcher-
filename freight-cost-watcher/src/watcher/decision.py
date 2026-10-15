"""Decision module: combine retrieval and classification to flag or justify rows."""
import logging
from datetime import date

from .notes import ParsedNote

logger = logging.getLogger(__name__)


def decide(
    route: str,
    week_monday: date,
    candidates: list[ParsedNote],
    is_valid_fn,
) -> tuple[str, str | None, list[str]]:
    """Decide whether a candidate row is justified or needs human review.

    Args:
        route: The route name.
        week_monday: The Monday of the week.
        candidates: Retrieved notes (route+date filtered, ranked by similarity).
        is_valid_fn: Function that takes a note text and returns bool.

    Returns:
        Tuple of (flagged, matched_note_id, retrieved_note_ids).
        - flagged: 'No (justified)' or 'Yes'
        - matched_note_id: note_id if justified, else ''
        - retrieved_note_ids: all retrieved note IDs for diagnostics
    """
    retrieved_ids = [n.note_id for n in candidates]

    # Filter to valid reasons
    valid = [n for n in candidates if is_valid_fn(n.text)]
    best = valid[0] if valid else None

    flagged = 'No (justified)' if best else 'Yes'
    matched_note_id = best.note_id if best else ''

    logger.debug(
        'Decision for %s/%s: %d candidates, %d valid -> %s (matched=%s)',
        route, week_monday, len(candidates), len(valid),
        flagged, matched_note_id or 'none',
    )

    return flagged, matched_note_id, retrieved_ids
