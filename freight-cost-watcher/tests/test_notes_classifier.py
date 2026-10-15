"""Tests for notes parsing and classifier (Phases 2 & 3)."""
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.watcher.notes import parse_notes, ParsedNote, _parse_end_date
from src.watcher.classifier import keyword_says_cost_up


# ── Note parsing tests ──

class TestNoteParsing:
    @pytest.fixture
    def notes_df(self):
        import pandas as pd
        return pd.read_csv(
            _PROJECT_ROOT / 'data' / 'context_notes.csv',
            parse_dates=['date'],
        )

    @pytest.fixture
    def parsed(self, notes_df):
        return parse_notes(notes_df)

    def test_ten_notes(self, parsed):
        assert len(parsed) == 10

    def test_n001_window(self, parsed):
        """N001: Feb 24 to Mar 8 (explicit range in text)."""
        n001 = next(n for n in parsed if n.note_id == 'N001')
        assert n001.start_date == date(2025, 2, 24)
        assert n001.end_date == date(2025, 3, 8)

    def test_n002_window(self, parsed):
        """N002: start + 6 days (no explicit range)."""
        n002 = next(n for n in parsed if n.note_id == 'N002')
        assert n002.start_date == date(2025, 1, 20)
        assert n002.end_date == date(2025, 1, 26)

    def test_n003_one_week_only(self, parsed):
        """N003 says 'starting this week'; must be treated as one week only.
        Judgment call: open-ended would wrongly justify routes after May 2025.
        """
        n003 = next(n for n in parsed if n.note_id == 'N003')
        assert n003.start_date == date(2025, 5, 5)
        assert n003.end_date == date(2025, 5, 11)

    def test_all_other_notes_default_window(self, parsed):
        """All notes except N001 should have end = start + 6."""
        for note in parsed:
            if note.note_id == 'N001':
                continue
            expected_end = note.start_date + timedelta(days=6)
            assert note.end_date == expected_end, (
                f'{note.note_id}: expected end {expected_end}, got {note.end_date}'
            )

    def test_overlap_n001_week_feb24(self, parsed):
        """N001 (Feb 24 - Mar 8) overlaps week of Feb 24."""
        n001 = next(n for n in parsed if n.note_id == 'N001')
        assert n001.overlaps_week(date(2025, 2, 24))

    def test_overlap_n001_week_mar3(self, parsed):
        """N001 (Feb 24 - Mar 8) overlaps week of Mar 3."""
        n001 = next(n for n in parsed if n.note_id == 'N001')
        assert n001.overlaps_week(date(2025, 3, 3))

    def test_no_overlap_n001_week_mar10(self, parsed):
        """N001 (Feb 24 - Mar 8) does NOT overlap week of Mar 10-16."""
        n001 = next(n for n in parsed if n.note_id == 'N001')
        assert not n001.overlaps_week(date(2025, 3, 10))

    def test_n003_does_not_overlap_week_may12(self, parsed):
        """N003 (May 5 - May 11) does NOT overlap week of May 12.
        This is critical: the Mumbai-Delhi May 12 row must stay Yes.
        """
        n003 = next(n for n in parsed if n.note_id == 'N003')
        assert not n003.overlaps_week(date(2025, 5, 12))


# ── Keyword classifier tests on the 10 real notes ──

class TestKeywordClassifierRealNotes:
    """Ground truth from spec section 5.1: N001-N003 accept, rest reject."""

    @pytest.fixture
    def notes_df(self):
        import pandas as pd
        return pd.read_csv(
            _PROJECT_ROOT / 'data' / 'context_notes.csv',
            parse_dates=['date'],
        )

    @pytest.fixture
    def parsed(self, notes_df):
        return parse_notes(notes_df)

    @pytest.fixture
    def note_map(self, parsed):
        return {n.note_id: n for n in parsed}

    def test_n001_accepts(self, note_map):
        """N001: flood, detours, higher trip costs → accept."""
        assert keyword_says_cost_up(note_map['N001'].text) == True

    def test_n002_accepts(self, note_map):
        """N002: festival week surcharge → accept."""
        assert keyword_says_cost_up(note_map['N002'].text) == True

    def test_n003_accepts(self, note_map):
        """N003: diesel rise, pushing up costs → accept."""
        assert keyword_says_cost_up(note_map['N003'].text) == True

    def test_n004_rejects(self, note_map):
        """N004: toll, but not in dataset → reject."""
        assert keyword_says_cost_up(note_map['N004'].text) == False

    def test_n005_rejects(self, note_map):
        """N005: maintenance, costs not significantly affected → reject."""
        assert keyword_says_cost_up(note_map['N005'].text) == False

    def test_n006_rejects(self, note_map):
        """N006: stable demand → reject."""
        assert keyword_says_cost_up(note_map['N006'].text) == False

    def test_n007_rejects(self, note_map):
        """N007: road improved → reject."""
        assert keyword_says_cost_up(note_map['N007'].text) == False

    def test_n008_rejects(self, note_map):
        """N008: normal operations → reject."""
        assert keyword_says_cost_up(note_map['N008'].text) == False

    def test_n009_rejects(self, note_map):
        """N009: back to normal after repairs → reject."""
        assert keyword_says_cost_up(note_map['N009'].text) == False

    def test_n010_rejects(self, note_map):
        """N010: tracking mandate, no rate change → reject."""
        assert keyword_says_cost_up(note_map['N010'].text) == False

    def test_all_10_correct(self, note_map):
        """All 10 notes classified correctly."""
        expected = {
            'N001': True, 'N002': True, 'N003': True,
            'N004': False, 'N005': False, 'N006': False,
            'N007': False, 'N008': False, 'N009': False, 'N010': False,
        }
        results = {nid: keyword_says_cost_up(n.text) for nid, n in note_map.items()}
        assert results == expected


# ── Tricky notes tests ──

class TestTrickyNotes:
    """eval/tricky_notes.csv: 10 crafted notes, all must be classified correctly."""

    TRICKY = [
        ('Costs may rise next month.', False),
        ('There was no increase in rates.', False),
        ('Costs rose sharply last year.', False),  # accept by keyword but date window fails
        ('Costs were higher.', False),  # no cause named
        ('The surcharge was waived this week.', False),
        ('Prices stayed flat despite the diesel increase.', False),
        ('A fuel surcharge raised rates 8% on this route.', True),
        ('Heavy rain forced detours, pushing up trip costs.', True),
        ('Transporters charged more due to a strike.', True),
        ('Costs went up because of a bridge closure.', True),
    ]

    @pytest.mark.parametrize('text,expected', TRICKY)
    def test_tricky_note(self, text, expected):
        result = keyword_says_cost_up(text)
        assert result == expected, (
            f'keyword_says_cost_up("{text}") = {result}, expected {expected}'
        )
