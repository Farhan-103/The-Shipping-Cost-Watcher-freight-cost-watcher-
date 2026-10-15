"""Flag route-weeks where the cost rise is unusual."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def add_flags(
    weekly: pd.DataFrame,
    flag_own_pct: float = 9.0,
    flag_peer_pct: float | None = None,
) -> pd.DataFrame:
    """Mark candidate rows where own-history rise exceeds the threshold.

    A row is a candidate if:
    - It has history (history_weeks_used > 0), AND
    - own_pct >= flag_own_pct.

    If flag_peer_pct is set, also flag when peer_pct >= flag_peer_pct
    (OR logic), but this is off by default because Mumbai-Pune is
    structurally pricier than its only Short peer Delhi-Jaipur.
    """
    weekly = weekly.copy()

    has_history = weekly['history_weeks_used'] > 0
    own_trigger = weekly['own_pct'] >= flag_own_pct

    if flag_peer_pct is not None:
        peer_trigger = weekly['peer_pct'] >= flag_peer_pct
        weekly['is_candidate'] = has_history & (own_trigger | peer_trigger)
    else:
        weekly['is_candidate'] = has_history & own_trigger

    n_candidates = weekly['is_candidate'].sum()
    logger.info(
        'Flagged %d candidates (own threshold=%.1f%%, peer threshold=%s)',
        n_candidates,
        flag_own_pct,
        f'{flag_peer_pct:.1f}%' if flag_peer_pct is not None else 'off',
    )

    return weekly
