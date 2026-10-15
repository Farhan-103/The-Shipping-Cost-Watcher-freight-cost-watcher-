"""Compute weekly cost-per-tonne-km for each (route, week_of)."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_weekly_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cost_per_tonne_km for each (route, week_of).

    Formula: sum(freight_cost_inr) / sum(quantity_tonnes * distance_km).
    Sum first, divide once — do NOT average per-trip ratios.

    Returns a DataFrame with columns: route, route_type, week_of, cost_per_tonne_km,
    sorted by route and week_of.
    """
    # Compute tonne_km for each shipment
    df = df.copy()
    df['tonne_km'] = df['quantity_tonnes'] * df['distance_km']

    # Group by route and week_of, summing cost and tonne_km
    grouped = (
        df.groupby(['route', 'route_type', 'week_of'], as_index=False)
        .agg(
            total_cost=('freight_cost_inr', 'sum'),
            total_tonne_km=('tonne_km', 'sum'),
            shipment_count=('shipment_id', 'count'),
        )
    )

    # Divide once: full precision
    grouped['cost_per_tonne_km'] = grouped['total_cost'] / grouped['total_tonne_km']

    # Sort deterministically
    grouped = grouped.sort_values(['route', 'week_of']).reset_index(drop=True)

    logger.info(
        'Computed weekly costs: %d route-weeks, cost range %.4f to %.4f',
        len(grouped),
        grouped['cost_per_tonne_km'].min(),
        grouped['cost_per_tonne_km'].max(),
    )

    return grouped[
        ['route', 'route_type', 'week_of', 'cost_per_tonne_km', 'shipment_count']
    ]
