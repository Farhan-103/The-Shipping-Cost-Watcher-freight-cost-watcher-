"""Compute baselines: own history and peer (similar-route) comparisons."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def add_own_history(
    weekly: pd.DataFrame,
    window: int = 8,
) -> pd.DataFrame:
    """Add own-history baseline: mean of up to `window` prior weeks for the same route.

    Rules:
    - Uses the calendar weeks strictly before the current week.
    - Week 1 of the data has no history (NaN).
    - Weeks 2-8 use 1-7 prior weeks (whatever is available).
    - Week 9+ uses exactly 8 prior weeks.
    - No look-ahead: only past data.
    """
    weekly = weekly.copy()
    weekly = weekly.sort_values(['route', 'week_of']).reset_index(drop=True)

    own_means = []
    history_weeks_used = []

    for route in sorted(weekly['route'].unique()):
        route_data = weekly[weekly['route'] == route].sort_values('week_of')
        values = route_data['cost_per_tonne_km'].values

        for i in range(len(route_data)):
            if i == 0:
                own_means.append(float('nan'))
                history_weeks_used.append(0)
            else:
                start = max(0, i - window)
                prior = values[start:i]
                own_means.append(prior.mean())
                history_weeks_used.append(len(prior))

    weekly['own_history_mean'] = own_means
    weekly['history_weeks_used'] = history_weeks_used
    weekly['history_weeks_used'] = weekly['history_weeks_used'].astype(int)

    return weekly


def add_peer_baseline(weekly: pd.DataFrame) -> pd.DataFrame:
    """Add peer baseline: mean of other routes with the same route_type in the same week.

    Excludes the route itself. Peer counts: Short=1, Medium=2, Long=1.
    """
    weekly = weekly.copy()

    type_week_mean = (
        weekly.groupby(['route_type', 'week_of'])['cost_per_tonne_km']
        .agg(['sum', 'count'])
        .reset_index()
    )
    type_week_mean.columns = ['route_type', 'week_of', 'type_sum', 'type_count']

    weekly = weekly.merge(type_week_mean, on=['route_type', 'week_of'], how='left')

    weekly['peer_count'] = weekly['type_count'] - 1
    weekly['peer_mean'] = (
        (weekly['type_sum'] - weekly['cost_per_tonne_km'])
        / weekly['peer_count']
    )

    weekly = weekly.drop(columns=['type_sum', 'type_count'])

    return weekly


def add_percentages(weekly: pd.DataFrame) -> pd.DataFrame:
    """Compute percentage differences vs own history and vs peers.

    percent = (value / baseline - 1) * 100
    Uses full precision; rounding happens only at display time.
    """
    weekly = weekly.copy()

    weekly['own_pct'] = (
        (weekly['cost_per_tonne_km'] / weekly['own_history_mean'] - 1) * 100
    )

    weekly['peer_pct'] = (
        (weekly['cost_per_tonne_km'] / weekly['peer_mean'] - 1) * 100
    )

    return weekly
