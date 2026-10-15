"""Golden regression test for the 9 candidate rows specified in Section 4."""
import sys
from pathlib import Path
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import config
from src.watcher.pipeline import run_pipeline


@pytest.fixture(scope="module")
def pipeline_results(tmp_path_factory):
    tmp_out = tmp_path_factory.mktemp("golden") / "output.csv"
    tmp_diag = tmp_path_factory.mktemp("golden") / "diag.csv"
    tmp_ret = tmp_path_factory.mktemp("golden") / "ret.csv"
    tmp_dis = tmp_path_factory.mktemp("golden") / "dis.csv"

    summary = run_pipeline(
        shipment_csv=config.SHIPMENT_CSV,
        notes_csv=config.NOTES_CSV,
        output_csv=tmp_out,
        diagnostics_csv=tmp_diag,
        retrieval_log=tmp_ret,
        disagreements_log=tmp_dis,
        flag_own_pct=config.FLAG_OWN_PCT,
        flag_peer_pct=config.FLAG_PEER_PCT,
        history_window=config.HISTORY_WINDOW,
        use_rag=False,  # deterministic filter
        use_llm_classifier=False,  # keyword-only offline
        use_llm_reasons=False,
        all_rows=False,
    )
    df = pd.read_csv(tmp_out, dtype=str).fillna("")
    return df


class TestGoldenRegression:
    def test_candidate_count(self, pipeline_results):
        """At 9.0% threshold, exactly 9 candidate rows are produced."""
        assert len(pipeline_results) == 9

    @pytest.mark.parametrize("route,week_of,expected_flagged,expected_note", [
        ("Delhi-Jaipur", "2024-11-11", "Yes", ""),
        ("Delhi-Jaipur", "2024-11-18", "Yes", ""),
        ("Ahmedabad-Mumbai", "2025-01-20", "No (justified)", "N002"),
        ("Chennai-Bangalore", "2025-02-24", "No (justified)", "N001"),
        ("Chennai-Bangalore", "2025-03-03", "No (justified)", "N001"),
        ("Chennai-Bangalore", "2025-03-10", "Yes", ""),
        ("Chennai-Bangalore", "2025-03-17", "Yes", ""),
        ("Mumbai-Delhi", "2025-05-12", "Yes", ""),
        ("Mumbai-Pune", "2025-09-15", "Yes", ""),
    ])
    def test_golden_row(self, pipeline_results, route, week_of, expected_flagged, expected_note):
        match = pipeline_results[
            (pipeline_results["route"] == route) &
            (pipeline_results["week_of"] == week_of)
        ]
        assert len(match) == 1, f"Missing row for {route} {week_of}"
        row = match.iloc[0]
        assert row["flagged"] == expected_flagged, (
            f"{route} {week_of}: expected flagged={expected_flagged}, got {row['flagged']}"
        )
        assert row["matched_note_id"] == expected_note, (
            f"{route} {week_of}: expected note={expected_note}, got {row['matched_note_id']}"
        )
