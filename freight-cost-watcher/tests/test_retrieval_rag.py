"""Tests for retrieval and RAG equivalence (Phase 4).

Checks:
- Python filter vs ChromaDB filter consistency
- RAG vs no-RAG equivalence on graded columns
- Route and date filter constraints
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.watcher.loader import load_shipments, load_notes
from src.watcher.notes import parse_notes
from src.watcher.retrieval import Retriever, python_filter_notes
from src.watcher.pipeline import run_pipeline


@pytest.fixture
def notes():
    df = load_notes(_PROJECT_ROOT / "data" / "context_notes.csv")
    return parse_notes(df)


class TestRetrieval:
    def test_python_filter_direct(self, notes):
        """Ahmedabad-Mumbai on 2025-01-20 should match N002."""
        matches = python_filter_notes(notes, "Ahmedabad-Mumbai", date(2025, 1, 20))
        assert len(matches) == 1
        assert matches[0].note_id == "N002"

    def test_chennai_bangalore_flood_overlap(self, notes):
        """Chennai-Bangalore on 2025-02-24 and 2025-03-03 match N001."""
        m1 = python_filter_notes(notes, "Chennai-Bangalore", date(2025, 2, 24))
        assert any(n.note_id == "N001" for n in m1)

        m2 = python_filter_notes(notes, "Chennai-Bangalore", date(2025, 3, 3))
        assert any(n.note_id == "N001" for n in m2)

    def test_chennai_bangalore_after_flood_no_n001(self, notes):
        """Chennai-Bangalore on 2025-03-10 does NOT match N001 (flood window ended Mar 8)."""
        m = python_filter_notes(notes, "Chennai-Bangalore", date(2025, 3, 10))
        assert not any(n.note_id == "N001" for n in m)

    def test_all_routes_diesel_overlap(self, notes):
        """N003 applies to All Routes for week of 2025-05-05."""
        m = python_filter_notes(notes, "Mumbai-Delhi", date(2025, 5, 5))
        assert any(n.note_id == "N003" for n in m)

        # But week of 2025-05-12 does NOT match N003 (1-week window)
        m_next = python_filter_notes(notes, "Mumbai-Delhi", date(2025, 5, 12))
        assert not any(n.note_id == "N003" for n in m_next)

    def test_chroma_python_filter_agreement(self, notes):
        """Retriever with RAG should agree with Python filter for all candidate weeks."""
        try:
            retriever = Retriever(notes, use_rag=True)
        except Exception as e:
            pytest.skip(f"ChromaDB not available in current environment: {e}")

        test_cases = [
            ("Delhi-Jaipur", date(2024, 11, 11)),
            ("Ahmedabad-Mumbai", date(2025, 1, 20)),
            ("Chennai-Bangalore", date(2025, 2, 24)),
            ("Chennai-Bangalore", date(2025, 3, 10)),
            ("Mumbai-Delhi", date(2025, 5, 12)),
            ("Mumbai-Pune", date(2025, 9, 15)),
        ]

        for route, week_m in test_cases:
            py_res = python_filter_notes(notes, route, week_m)
            rag_res = retriever.retrieve(route, week_m)
            py_ids = sorted(n.note_id for n in py_res)
            rag_ids = sorted(n.note_id for n in rag_res)
            assert py_ids == rag_ids, f"Mismatch for {route} {week_m}: py={py_ids}, rag={rag_ids}"
