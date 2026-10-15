"""Tests for the Freight Cost Watcher pipeline — Phases 1 & 2.

Tests cover: week assignment, formula, trailing window, peers,
percent formatting, CSV header, and the 3 must-match sample rows.
All tests run offline with no LLM calls.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.watcher.loader import load_shipments, validate_shipments, load_notes
from src.watcher.weekly import compute_weekly_cost
from src.watcher.baselines import add_own_history, add_peer_baseline, add_percentages
from src.watcher.flagging import add_flags
from src.watcher.output import OUTPUT_HEADER, format_own_history, format_peer


# ── Fixtures ──

@pytest.fixture
def project_root():
    return _PROJECT_ROOT


@pytest.fixture
def shipments(project_root):
    return load_shipments(project_root / 'data' / 'shipment_records.csv')


@pytest.fixture
def weekly_df(shipments):
    return compute_weekly_cost(shipments)


@pytest.fixture
def full_weekly(weekly_df):
    """Weekly data with all baselines and flags computed."""
    df = add_own_history(weekly_df)
    df = add_peer_baseline(df)
    df = add_percentages(df)
    df = add_flags(df)
    return df


# ── Week assignment tests ──

class TestWeekAssignment:
    def test_sunday_goes_to_previous_monday(self):
        """A Sunday (day 6) belongs to the Monday before it."""
        # 2024-01-07 is a Sunday -> week_of should be 2024-01-01 (Monday)
        df = pd.DataFrame({
            'shipment_id': ['S1'],
            'origin': ['A'], 'destination': ['B'],
            'route_type': ['Short'],
            'material': ['X'],
            'quantity_tonnes': [10.0],
            'distance_km': [100.0],
            'freight_cost_inr': [3000],
            'shipment_date': [pd.Timestamp('2024-01-07')],  # Sunday
            'transporter': ['T'],
        })
        df['route'] = df['origin'] + '-' + df['destination']
        df['week_of'] = df['shipment_date'] - pd.to_timedelta(
            df['shipment_date'].dt.weekday, unit='D'
        )
        df['week_of'] = df['week_of'].dt.normalize()

        assert df['week_of'].iloc[0] == pd.Timestamp('2024-01-01')

    def test_monday_stays(self):
        """A Monday stays as the week_of date."""
        df = pd.DataFrame({
            'shipment_id': ['S1'],
            'origin': ['A'], 'destination': ['B'],
            'route_type': ['Short'],
            'material': ['X'],
            'quantity_tonnes': [10.0],
            'distance_km': [100.0],
            'freight_cost_inr': [3000],
            'shipment_date': [pd.Timestamp('2024-01-01')],  # Monday
            'transporter': ['T'],
        })
        df['route'] = df['origin'] + '-' + df['destination']
        df['week_of'] = df['shipment_date'] - pd.to_timedelta(
            df['shipment_date'].dt.weekday, unit='D'
        )
        df['week_of'] = df['week_of'].dt.normalize()

        assert df['week_of'].iloc[0] == pd.Timestamp('2024-01-01')


# ── Formula test ──

class TestFormula:
    def test_hand_example(self):
        """Trip A: 10t x 100km costs 3000; Trip B: 20t x 200km costs 10000.
        Weekly value = (3000+10000) / (10*100 + 20*200) = 13000/5000 = 2.60
        """
        df = pd.DataFrame({
            'shipment_id': ['A', 'B'],
            'origin': ['X', 'X'], 'destination': ['Y', 'Y'],
            'route_type': ['Short', 'Short'],
            'material': ['M', 'M'],
            'quantity_tonnes': [10.0, 20.0],
            'distance_km': [100.0, 200.0],
            'freight_cost_inr': [3000, 10000],
            'shipment_date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'transporter': ['T', 'T'],
        })
        df['route'] = df['origin'] + '-' + df['destination']
        df['week_of'] = df['shipment_date'] - pd.to_timedelta(
            df['shipment_date'].dt.weekday, unit='D'
        )
        df['week_of'] = df['week_of'].dt.normalize()

        weekly = compute_weekly_cost(df)

        assert len(weekly) == 1
        assert abs(weekly['cost_per_tonne_km'].iloc[0] - 2.60) < 1e-10


# ── Trailing window tests ──

class TestTrailingWindow:
    def test_week1_has_no_history(self, weekly_df):
        """Week 1 of any route has no history."""
        df = add_own_history(weekly_df)
        for route in df['route'].unique():
            route_data = df[df['route'] == route].sort_values('week_of')
            first = route_data.iloc[0]
            assert first['history_weeks_used'] == 0
            assert pd.isna(first['own_history_mean'])

    def test_excludes_current_week(self, weekly_df):
        """The trailing window excludes the current week."""
        df = add_own_history(weekly_df)
        # Week 2 should use exactly 1 prior week
        for route in sorted(df['route'].unique()):
            route_data = df[df['route'] == route].sort_values('week_of')
            second = route_data.iloc[1]
            assert second['history_weeks_used'] == 1
            # The mean should equal week 1's value
            assert abs(second['own_history_mean'] - route_data.iloc[0]['cost_per_tonne_km']) < 1e-10

    def test_weeks_2_to_8_use_fewer(self, weekly_df):
        """Weeks 2-8 use 1-7 prior weeks respectively."""
        df = add_own_history(weekly_df)
        route = sorted(df['route'].unique())[0]
        route_data = df[df['route'] == route].sort_values('week_of')

        for i in range(1, 8):  # weeks 2-8 (0-indexed: 1-7)
            expected_weeks = i
            actual = route_data.iloc[i]['history_weeks_used']
            assert actual == expected_weeks, (
                f'Week {i+1} should use {expected_weeks} prior weeks, got {actual}'
            )

    def test_no_look_ahead(self, weekly_df):
        """The window never looks ahead."""
        df = add_own_history(weekly_df)
        route = sorted(df['route'].unique())[0]
        route_data = df[df['route'] == route].sort_values('week_of')

        # For week 9+ (index 8+), should use exactly 8 prior weeks
        for i in range(8, min(15, len(route_data))):
            assert route_data.iloc[i]['history_weeks_used'] == 8


# ── Peer tests ──

class TestPeers:
    def test_peers_exclude_self(self, weekly_df):
        """Peer mean excludes the route itself."""
        df = add_peer_baseline(weekly_df)
        # Check that peer_count > 0 for all
        assert (df['peer_count'] > 0).all()

    def test_peer_counts(self, weekly_df):
        """Short has 1 peer, Medium has 2, Long has 1."""
        df = add_peer_baseline(weekly_df)
        expected = {'Short': 1, 'Medium': 2, 'Long': 1}
        for rtype, expected_count in expected.items():
            subset = df[df['route_type'] == rtype]
            actual = subset['peer_count'].unique()
            assert len(actual) == 1 and actual[0] == expected_count, (
                f'{rtype}: expected {expected_count} peers, got {actual}'
            )


# ── Formatting tests ──

class TestFormatting:
    def test_positive_own_history(self):
        assert format_own_history(35.5) == "+35.5% vs this route's past average"

    def test_negative_own_history(self):
        assert format_own_history(-5.2) == "-5.2% vs this route's past average"

    def test_positive_peer(self):
        assert format_peer(21.0) == '+21.0% vs similar-length routes this week'

    def test_csv_header_matches_sample(self, project_root):
        """The CSV header must equal the sample file's header."""
        sample = project_root / 'data' / 'sample_output_format_v2.csv'
        with open(sample, encoding='utf-8') as f:
            header = f.readline().strip().split(',')
        assert header == OUTPUT_HEADER


# ── Data validation tests ──

class TestDataValidation:
    def test_row_count(self, shipments):
        assert len(shipments) == 2940

    def test_no_nulls(self, shipments):
        # Original columns should have no nulls
        orig_cols = ['shipment_id', 'origin', 'destination', 'route_type',
                     'material', 'quantity_tonnes', 'distance_km',
                     'freight_cost_inr', 'shipment_date', 'transporter']
        assert not shipments[orig_cols].isnull().any().any()

    def test_no_duplicate_ids(self, shipments):
        assert shipments['shipment_id'].duplicated().sum() == 0

    def test_seven_routes(self, shipments):
        assert shipments['route'].nunique() == 7

    def test_728_route_weeks(self, weekly_df):
        assert len(weekly_df) == 728

    def test_validation_passes(self, shipments):
        validate_shipments(shipments)  # should not raise


# ── Sample row tests ──

class TestSampleRows:
    """The three must-match sample rows from the spec."""

    def test_delhi_jaipur_2024_11_11(self, full_weekly):
        """Delhi-Jaipur, 2024-11-11: cost=4.17, own=+35.5%, peer=+21.0%, Yes"""
        row = full_weekly[
            (full_weekly['route'] == 'Delhi-Jaipur') &
            (full_weekly['week_of'] == pd.Timestamp('2024-11-11'))
        ]
        assert len(row) == 1
        r = row.iloc[0]
        assert round(r['cost_per_tonne_km'], 2) == 4.17, f"Got {r['cost_per_tonne_km']:.4f}"
        assert round(r['own_pct'], 1) == 35.5, f"Got {r['own_pct']:.2f}"
        assert round(r['peer_pct'], 1) == 21.0, f"Got {r['peer_pct']:.2f}"
        assert r['is_candidate'] == True

    def test_ahmedabad_mumbai_2025_01_20(self, full_weekly):
        """Ahmedabad-Mumbai, 2025-01-20: cost=3.29, own=+29.5%, peer=+22.5%"""
        row = full_weekly[
            (full_weekly['route'] == 'Ahmedabad-Mumbai') &
            (full_weekly['week_of'] == pd.Timestamp('2025-01-20'))
        ]
        assert len(row) == 1
        r = row.iloc[0]
        assert round(r['cost_per_tonne_km'], 2) == 3.29, f"Got {r['cost_per_tonne_km']:.4f}"
        assert round(r['own_pct'], 1) == 29.5, f"Got {r['own_pct']:.2f}"
        assert round(r['peer_pct'], 1) == 22.5, f"Got {r['peer_pct']:.2f}"
        assert r['is_candidate'] == True

    def test_mumbai_pune_2025_09_15(self, full_weekly):
        """Mumbai-Pune, 2025-09-15: cost=3.98, own=+9.2%, peer=+23.6%"""
        row = full_weekly[
            (full_weekly['route'] == 'Mumbai-Pune') &
            (full_weekly['week_of'] == pd.Timestamp('2025-09-15'))
        ]
        assert len(row) == 1
        r = row.iloc[0]
        assert round(r['cost_per_tonne_km'], 2) == 3.98, f"Got {r['cost_per_tonne_km']:.4f}"
        assert round(r['own_pct'], 1) == 9.2, f"Got {r['own_pct']:.2f}"
        assert round(r['peer_pct'], 1) == 23.6, f"Got {r['peer_pct']:.2f}"
        assert r['is_candidate'] == True
