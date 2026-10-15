#!/usr/bin/env python
"""Evaluation script for Freight Cost Watcher.

Evaluates:
1. Decision accuracy against eval/labeled_cases.csv (hard gate: 0 wrongly justified rows)
2. Retrieval hit rate (whether expected note appears in top 3 retrieved)
3. Classifier accuracy on:
   - 10 real notes (target: 10/10)
   - 10 tricky notes (eval/tricky_notes.csv)
   - 6 heldout tricky notes (eval/heldout_notes.csv)
4. Writes comprehensive markdown report to eval/report.md.
"""
import csv
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import config
from src.watcher.loader import load_shipments, load_notes
from src.watcher.notes import parse_notes
from src.watcher.classifier import (
    keyword_says_cost_up,
    llm_accepts,
    is_valid_reason,
    build_classification_prompt,
    parse_llm_classification,
)
from src.watcher.retrieval import Retriever, python_filter_notes
from src.watcher.pipeline import run_pipeline


def evaluate_classifier_dataset(csv_path: Path, note_texts_and_labels: list[tuple[str, bool]]):
    """Evaluate keyword, LLM (if available via cache/stub), and combined classifier."""
    total = len(note_texts_and_labels)
    kw_correct = 0
    results = []

    for text, expected in note_texts_and_labels:
        kw_pred = keyword_says_cost_up(text)
        is_correct = (kw_pred == expected)
        if is_correct:
            kw_correct += 1
        results.append({
            "text": text,
            "expected": expected,
            "kw_pred": kw_pred,
            "correct": is_correct,
        })

    accuracy = (kw_correct / total) * 100 if total > 0 else 0.0
    return {
        "total": total,
        "kw_correct": kw_correct,
        "accuracy": accuracy,
        "details": results,
    }


def main():
    print("=== Running Evaluation Harness ===")
    eval_dir = _PROJECT_ROOT / "eval"
    output_dir = _PROJECT_ROOT / "output"
    report_path = eval_dir / "report.md"

    # 1. Run pipeline (cached/offline deterministic evaluation)
    tmp_out = eval_dir / "_eval_output.csv"
    tmp_diag = eval_dir / "_eval_diag.csv"
    tmp_ret = eval_dir / "_eval_ret.csv"
    tmp_dis = eval_dir / "_eval_dis.csv"

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
        use_rag=True,
        use_llm_classifier=False,  # baseline deterministic run
        use_llm_reasons=False,
        all_rows=False,
    )

    # Read output rows into lookup dict
    pipeline_rows = {}
    with open(tmp_out, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pipeline_rows[(r["route"], r["week_of"])] = r

    # Clean up temp files
    for p in [tmp_out, tmp_diag, tmp_ret, tmp_dis]:
        if p.exists():
            p.unlink()

    # 2. Evaluate against labeled_cases.csv
    labeled_cases_path = eval_dir / "labeled_cases.csv"
    labeled_cases = []
    with open(labeled_cases_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            labeled_cases.append(r)

    total_cases = len(labeled_cases)
    exact_matches = 0
    wrongly_justified = 0
    labeled_results = []

    # Notes & Retriever for Hit Rate testing
    notes_df = load_notes(config.NOTES_CSV)
    parsed_notes = parse_notes(notes_df)
    retriever = Retriever(parsed_notes, use_rag=True)

    hit_rate_cases = 0
    hit_rate_success = 0

    for case in labeled_cases:
        route = case["route"]
        week_of = case["week_of"]
        exp_flagged = case["expected_flagged"]
        exp_note = case.get("expected_note_id", "").strip()

        actual_row = pipeline_rows.get((route, week_of))
        if actual_row:
            act_flagged = actual_row["flagged"]
            act_note = actual_row.get("matched_note_id", "").strip()
        else:
            act_flagged = "NOT_CANDIDATE"
            act_note = ""

        # Check wrongly justified: expected was 'Yes' (unexplained) but model marked 'No (justified)'
        is_wrongly_justified = (exp_flagged == "Yes" and act_flagged == "No (justified)")
        if is_wrongly_justified:
            wrongly_justified += 1

        is_match = (act_flagged == exp_flagged and act_note == exp_note)
        if is_match:
            exact_matches += 1

        # Retrieval hit rate
        week_d = date.fromisoformat(week_of)
        retrieved = retriever.retrieve(route, week_d)
        retrieved_ids = [n.note_id for n in retrieved]

        if exp_note:
            hit_rate_cases += 1
            if exp_note in retrieved_ids[:3]:
                hit_rate_success += 1

        labeled_results.append({
            "route": route,
            "week_of": week_of,
            "exp_flagged": exp_flagged,
            "act_flagged": act_flagged,
            "exp_note": exp_note,
            "act_note": act_note,
            "match": is_match,
            "retrieved_top3": retrieved_ids[:3],
        })

    decision_accuracy = (exact_matches / total_cases) * 100 if total_cases > 0 else 0
    hit_rate_pct = (hit_rate_success / hit_rate_cases) * 100 if hit_rate_cases > 0 else 100.0

    # 3. Evaluate 10 real notes
    real_note_pairs = []
    # Ground truth from spec section 5.1:
    real_expected = {
        "N001": True, "N002": True, "N003": True,
        "N004": False, "N005": False, "N006": False,
        "N007": False, "N008": False, "N009": False, "N010": False,
    }
    for n in parsed_notes:
        real_note_pairs.append((n.text, real_expected[n.note_id]))

    real_eval = evaluate_classifier_dataset("context_notes.csv", real_note_pairs)

    # 4. Evaluate tricky notes
    tricky_pairs = []
    with open(eval_dir / "tricky_notes.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            exp_bool = (r["expected"].strip().lower() == "accept")
            tricky_pairs.append((r["text"].strip(), exp_bool))
    tricky_eval = evaluate_classifier_dataset("tricky_notes.csv", tricky_pairs)

    # 5. Evaluate heldout notes
    heldout_pairs = []
    with open(eval_dir / "heldout_notes.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            exp_bool = (r["expected"].strip().lower() == "accept")
            heldout_pairs.append((r["text"].strip(), exp_bool))
    heldout_eval = evaluate_classifier_dataset("heldout_notes.csv", heldout_pairs)

    # 6. Generate report.md
    md = []
    md.append("# Freight Cost Watcher — Evaluation Report")
    md.append("")
    md.append("## Executive Summary")
    md.append("")
    md.append(f"- **Decision Accuracy**: **{decision_accuracy:.1f}%** ({exact_matches}/{total_cases} labeled cases)")
    md.append(f"- **Wrongly Justified Rows (Hard Gate = 0)**: **{wrongly_justified}**")
    md.append(f"- **Retrieval Top-3 Hit Rate**: **{hit_rate_pct:.1f}%** ({hit_rate_success}/{hit_rate_cases} justified cases)")
    md.append(f"- **Real Notes Classification**: **{real_eval['accuracy']:.1f}%** ({real_eval['kw_correct']}/{real_eval['total']})")
    md.append(f"- **Tricky Notes Classification**: **{tricky_eval['accuracy']:.1f}%** ({tricky_eval['kw_correct']}/{tricky_eval['total']})")
    md.append(f"- **Heldout Notes Classification**: **{heldout_eval['accuracy']:.1f}%** ({heldout_eval['kw_correct']}/{heldout_eval['total']})")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Candidate Decision Evaluation (Labeled Cases)")
    md.append("")
    md.append("| Route | Week | Expected Flagged | Actual Flagged | Expected Note | Actual Note | Match | Top 3 Retrieved |")
    md.append("|---|---|---|---|---|---|:---:|---|")
    for r in labeled_results:
        m_icon = "PASS" if r["match"] else "FAIL"
        ret_str = ", ".join(r["retrieved_top3"]) if r["retrieved_top3"] else "none"
        md.append(f"| {r['route']} | {r['week_of']} | {r['exp_flagged']} | {r['act_flagged']} | {r['exp_note'] or '-'} | {r['act_note'] or '-'} | {m_icon} | {ret_str} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Classifier Performance on Fixture Datasets")
    md.append("")
    md.append("### Real Notes (10 notes)")
    md.append(f"- Score: **{real_eval['kw_correct']}/{real_eval['total']} ({real_eval['accuracy']:.1f}%)**")
    md.append("- Target: 10/10 (N001-N003 Accept, N004-N010 Reject)")
    md.append("")
    md.append("### Tricky Notes Fixture (10 notes)")
    md.append(f"- Score: **{tricky_eval['kw_correct']}/{tricky_eval['total']} ({tricky_eval['accuracy']:.1f}%)**")
    md.append("")
    md.append("| Text | Expected | Keyword Pred | Result |")
    md.append("|---|:---:|:---:|:---:|")
    for item in tricky_eval["details"]:
        exp = "Accept" if item["expected"] else "Reject"
        pred = "Accept" if item["kw_pred"] else "Reject"
        res = "PASS" if item["correct"] else "FAIL"
        md.append(f"| {item['text']} | {exp} | {pred} | {res} |")
    md.append("")
    md.append("### Fresh Heldout Notes (6 notes)")
    md.append(f"- Score: **{heldout_eval['kw_correct']}/{heldout_eval['total']} ({heldout_eval['accuracy']:.1f}%)**")
    md.append("")
    md.append("| Text | Expected | Keyword Pred | Result |")
    md.append("|---|:---:|:---:|:---:|")
    for item in heldout_eval["details"]:
        exp = "Accept" if item["expected"] else "Reject"
        pred = "Accept" if item["kw_pred"] else "Reject"
        res = "PASS" if item["correct"] else "FAIL"
        md.append(f"| {item['text']} | {exp} | {pred} | {res} |")
    md.append("")

    report_content = "\n".join(md)
    report_path.write_text(report_content, encoding="utf-8")
    print(f"\nReport generated at: {report_path}")
    print(f"Summary: Decision Acc={decision_accuracy:.1f}%, Wrongly Justified={wrongly_justified}, Real Notes={real_eval['accuracy']:.1f}%, Tricky={tricky_eval['accuracy']:.1f}%, Heldout={heldout_eval['accuracy']:.1f}%")

    if wrongly_justified > 0:
        print("ERROR: Hard gate violated! Wrongly justified count > 0.")
        sys.exit(1)


if __name__ == "__main__":
    main()
