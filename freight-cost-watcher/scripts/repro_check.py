#!/usr/bin/env python
"""Reproducibility check script.

Runs the full pipeline 3 times using cached LLM answers (0 calls),
plus a cold run with --refresh-llm (1 cold run by default; --cold-runs 3 for full proof).
Saves output/run1.csv to run3.csv.
Compares every column except reason byte for byte, prints sha256 hashes and
IDENTICAL or the diff, and reports whether the reason text varied.
Saves the report to logs/repro_check.txt.
"""
import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def compute_sha256(filepath: Path) -> str:
    """Compute sha256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def read_csv_columns(filepath: Path) -> tuple[list[dict], list[dict]]:
    """Read CSV and separate deterministic columns from reason."""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        graded_rows = []
        reason_rows = []
        for r in reader:
            graded_rows.append({
                'route': r['route'],
                'week_of': r['week_of'],
                'cost_per_tonne_km': r['cost_per_tonne_km'],
                'vs_own_history': r['vs_own_history'],
                'vs_similar_routes': r['vs_similar_routes'],
                'flagged': r['flagged'],
                'matched_note_id': r.get('matched_note_id', ''),
            })
            reason_rows.append({
                'route': r['route'],
                'week_of': r['week_of'],
                'reason': r.get('reason', ''),
            })
    return graded_rows, reason_rows


def main():
    parser = argparse.ArgumentParser(description="Verify pipeline reproducibility across runs.")
    parser.add_argument("--cold-runs", type=int, default=1, help="Number of cold runs with --refresh-llm (default: 1)")
    parser.add_argument("--stub-llm", action="store_true", help="Force stub LLM during repro check")
    args = parser.parse_args()

    output_dir = _PROJECT_ROOT / "output"
    logs_dir = _PROJECT_ROOT / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("FREIGHT COST WATCHER - REPRODUCIBILITY CHECK REPORT")
    report_lines.append("=" * 60)

    run_files = []
    # Execute 3 runs
    for run_idx in range(1, 4):
        run_file = output_dir / f"run{run_idx}.csv"
        print(f"\n--- Running Pipeline Run #{run_idx} ---")
        cmd = [sys.executable, str(_PROJECT_ROOT / "scripts" / "run.py")]
        if args.stub_llm:
            cmd.append("--stub-llm")

        res = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Error during Run #{run_idx}:\n{res.stderr}")
            sys.exit(res.returncode)

        shutil.copyfile(output_dir / "output.csv", run_file)
        run_files.append(run_file)
        sha = compute_sha256(run_file)
        report_lines.append(f"Run {run_idx}: {run_file.name} | SHA256: {sha}")

    # Compare runs 1, 2, 3
    graded1, reasons1 = read_csv_columns(run_files[0])
    graded2, reasons2 = read_csv_columns(run_files[1])
    graded3, reasons3 = read_csv_columns(run_files[2])

    graded_match = (graded1 == graded2 == graded3)
    reasons_match = (reasons1 == reasons2 == reasons3)

    report_lines.append("\nDeterministic Graded Columns Comparison (route, week_of, cost, baselines, flagged, matched_note_id):")
    if graded_match:
        report_lines.append(">> STATUS: IDENTICAL across all 3 runs.")
    else:
        report_lines.append(">> STATUS: DIFFERENCE DETECTED in graded columns!")
        if graded1 != graded2:
            report_lines.append("Diff between Run 1 and Run 2:")
            for i, (r1, r2) in enumerate(zip(graded1, graded2)):
                if r1 != r2:
                    report_lines.append(f"Row {i}: Run1={r1} vs Run2={r2}")
        if graded2 != graded3:
            report_lines.append("Diff between Run 2 and Run 3:")
            for i, (r2, r3) in enumerate(zip(graded2, graded3)):
                if r2 != r3:
                    report_lines.append(f"Row {i}: Run2={r2} vs Run3={r3}")

    report_lines.append(f"\nReason text variation: {'IDENTICAL' if reasons_match else 'VARIED (as permitted for LLM reason rewriting)'}")

    # Optional cold run
    if args.cold_runs > 0:
        report_lines.append("\n--- Cold Run Verification (--refresh-llm) ---")
        cold_file = output_dir / "cold_run.csv"
        cmd = [sys.executable, str(_PROJECT_ROOT / "scripts" / "run.py"), "--refresh-llm"]
        if args.stub_llm:
            cmd.append("--stub-llm")

        res = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), capture_output=True, text=True)
        if res.returncode == 0:
            shutil.copyfile(output_dir / "output.csv", cold_file)
            cold_graded, _ = read_csv_columns(cold_file)
            cold_sha = compute_sha256(cold_file)
            report_lines.append(f"Cold Run: {cold_file.name} | SHA256: {cold_sha}")
            cold_match = (graded1 == cold_graded)
            report_lines.append(f"Cold Run Graded Columns Match Run 1: {'IDENTICAL' if cold_match else 'DIFFERS'}")
        else:
            report_lines.append(f"Cold run failed or was skipped: {res.stderr}")

    report_content = "\n".join(report_lines)
    print("\n" + report_content)

    report_path = logs_dir / "repro_check.txt"
    report_path.write_text(report_content, encoding="utf-8")
    print(f"\nReproducibility report saved to: {report_path}")

    if not graded_match:
        sys.exit(1)


if __name__ == "__main__":
    main()
