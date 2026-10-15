"""Load and validate input CSV files."""
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataFacts:
    """Expected facts about the shipment data, asserted on load."""
    total_rows: int = 2940
    num_routes: int = 7
    num_weeks_per_route: int = -1  # set dynamically
    total_route_weeks: int = 728
    weeks_with_history: int = 721
    expected_routes: list[str] = field(default_factory=lambda: sorted([
        'Mumbai-Pune', 'Delhi-Jaipur', 'Ahmedabad-Mumbai',
        'Chennai-Bangalore', 'Kolkata-Bhubaneswar',
        'Mumbai-Delhi', 'Delhi-Chennai',
    ]))


def load_shipments(path: Path) -> pd.DataFrame:
    """Load shipment_records.csv, parse dates, add route and week_of columns."""
    df = pd.read_csv(path, parse_dates=['shipment_date'])
    
    # Validate no nulls
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning('Null values found:\n%s', null_counts[null_counts > 0])
    assert not df.isnull().any().any(), 'Input data contains null values'
    
    # Validate no duplicate shipment_id
    dupes = df['shipment_id'].duplicated().sum()
    if dupes > 0:
        logger.warning('%d duplicate shipment_ids found', dupes)
    assert dupes == 0, f'{dupes} duplicate shipment_ids found'
    
    # Build route column
    df['route'] = df['origin'] + '-' + df['destination']
    
    # Compute week_of: Monday of the ISO week containing shipment_date
    # Monday = 0, Sunday = 6; subtract weekday to get Monday
    df['week_of'] = df['shipment_date'] - pd.to_timedelta(
        df['shipment_date'].dt.weekday, unit='D'
    )
    df['week_of'] = df['week_of'].dt.normalize()  # strip time component
    
    return df


def validate_shipments(df: pd.DataFrame) -> None:
    """Assert expected facts about the data and warn if different."""
    facts = DataFacts()
    
    n_rows = len(df)
    if n_rows != facts.total_rows:
        logger.warning('Expected %d rows, got %d', facts.total_rows, n_rows)
    assert n_rows == facts.total_rows, f'Expected {facts.total_rows} rows, got {n_rows}'
    
    routes = sorted(df['route'].unique().tolist())
    if routes != facts.expected_routes:
        logger.warning('Routes differ. Got: %s', routes)
    assert routes == facts.expected_routes, f'Expected routes {facts.expected_routes}, got {routes}'
    
    # Check route-week combinations
    route_weeks = df.groupby(['route', 'week_of']).size().reset_index(name='count')
    n_route_weeks = len(route_weeks)
    if n_route_weeks != facts.total_route_weeks:
        logger.warning('Expected %d route-weeks, got %d', facts.total_route_weeks, n_route_weeks)
    assert n_route_weeks == facts.total_route_weeks, (
        f'Expected {facts.total_route_weeks} route-weeks, got {n_route_weeks}'
    )
    
    # Verify 3-5 shipments per route-week
    min_count = route_weeks['count'].min()
    max_count = route_weeks['count'].max()
    if min_count < 3 or max_count > 5:
        logger.warning('Shipments per route-week range: %d to %d (expected 3-5)', min_count, max_count)
    
    # Every route should have data in all weeks
    weeks_per_route = df.groupby('route')['week_of'].nunique()
    n_weeks = weeks_per_route.iloc[0]
    expected_weeks = facts.total_route_weeks // facts.num_routes  # 104
    assert (weeks_per_route == expected_weeks).all(), (
        f'Not all routes have {expected_weeks} weeks: {weeks_per_route.to_dict()}'
    )
    
    logger.info(
        'Data validated: %d rows, %d routes, %d route-weeks, '
        '%d-%d shipments per route-week',
        n_rows, len(routes), n_route_weeks, min_count, max_count,
    )


def load_notes(path: Path) -> pd.DataFrame:
    """Load context_notes.csv and parse dates."""
    df = pd.read_csv(path, parse_dates=['date'])
    assert len(df) == 10, f'Expected 10 notes, got {len(df)}'
    assert not df.isnull().any().any(), 'Notes contain null values'
    return df
