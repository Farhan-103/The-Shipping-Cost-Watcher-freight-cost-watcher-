"""Pipeline orchestrator: runs all stages in order."""
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from . import loader, weekly, baselines, flagging, notes
from . import classifier, retrieval, decision, reasons, output
from .llm_client import BaseLLMClient, create_llm_client
from .notes import ParsedNote

logger = logging.getLogger(__name__)


def run_pipeline(
    shipment_csv: Path,
    notes_csv: Path,
    output_csv: Path,
    diagnostics_csv: Path,
    retrieval_log: Path,
    disagreements_log: Path,
    flag_own_pct: float = 9.0,
    flag_peer_pct: float | None = None,
    history_window: int = 8,
    use_rag: bool = True,
    use_llm_classifier: bool = True,
    use_llm_reasons: bool = False,
    all_rows: bool = False,
    llm_client: BaseLLMClient | None = None,
    refresh_llm: bool = False,
) -> dict:
    """Run the full analysis pipeline.

    Returns a summary dict with key stats.
    """
    logger.info('=== Starting Freight Cost Watcher Pipeline ===')

    # ── Phase 1: Load and validate data ──
    logger.info('Phase 1: Loading data...')
    shipments = loader.load_shipments(shipment_csv)
    loader.validate_shipments(shipments)

    notes_df = loader.load_notes(notes_csv)
    parsed_notes = notes.parse_notes(notes_df)

    # ── Phase 2: Compute weekly costs and baselines ──
    logger.info('Phase 2: Computing weekly costs and baselines...')
    weekly_df = weekly.compute_weekly_cost(shipments)
    weekly_df = baselines.add_own_history(weekly_df, window=history_window)
    weekly_df = baselines.add_peer_baseline(weekly_df)
    weekly_df = baselines.add_percentages(weekly_df)

    # ── Phase 3: Flag candidates ──
    logger.info('Phase 3: Flagging candidates...')
    weekly_df = flagging.add_flags(weekly_df, flag_own_pct, flag_peer_pct)

    # ── Phase 4: LLM classification of notes ──
    llm_results: dict[str, dict | None] = {}
    if use_llm_classifier and llm_client is not None:
        logger.info('Phase 4: Classifying notes with LLM...')
        for note in parsed_notes:
            prompt = classifier.build_classification_prompt(note.text)
            try:
                response = llm_client.complete(
                    prompt,
                    purpose='note_classification',
                    max_tokens=100,
                    bypass_cache=refresh_llm,
                )
                parsed = classifier.parse_llm_classification(response.text)
                llm_results[note.note_id] = parsed

                # Log disagreements
                kw = classifier.keyword_says_cost_up(note.text)
                llm_ok = classifier.llm_accepts(parsed) if parsed else False
                if kw != llm_ok:
                    classifier.log_disagreement(
                        disagreements_log, note.note_id, kw, llm_ok, note.text,
                    )
            except Exception as e:
                logger.error('LLM classification failed for %s: %s', note.note_id, e)
                llm_results[note.note_id] = None
    else:
        logger.info('Phase 4: Skipping LLM classification (using keyword-only)')

    # ── Phase 5: Set up retriever ──
    logger.info('Phase 5: Setting up retrieval...')
    retriever = retrieval.Retriever(
        parsed_notes, use_rag=use_rag, log_path=retrieval_log,
    )

    # Build validity check function
    def is_valid_fn(note_text: str) -> bool:
        """Check if a note is a valid reason using configured checks."""
        if use_llm_classifier and llm_client is not None:
            # Find the note by text to get its LLM result
            for note in parsed_notes:
                if note.text == note_text:
                    llm_result = llm_results.get(note.note_id)
                    return classifier.is_valid_reason(note_text, llm_result)
            return classifier.is_valid_reason(note_text, keyword_only=True)
        return classifier.is_valid_reason(note_text, keyword_only=True)

    # ── Phase 6: Process each row ──
    logger.info('Phase 6: Processing rows and generating decisions...')
    all_note_ids = [n.note_id for n in parsed_notes]
    output_rows = []
    diagnostics_rows = []

    # Process in sorted order for determinism
    for _, row in weekly_df.sort_values(['route', 'week_of']).iterrows():
        route = row['route']
        week_of = row['week_of']
        week_monday = week_of.date() if hasattr(week_of, 'date') else week_of
        cost = row['cost_per_tonne_km']
        is_candidate = row['is_candidate']

        # Format baseline strings
        has_history = row['history_weeks_used'] > 0
        own_pct = row['own_pct'] if has_history else float('nan')
        peer_pct = row['peer_pct']

        vs_own = output.format_own_history(own_pct) if has_history else ''
        vs_peer = output.format_peer(peer_pct)
        own_pct_str = f"{'+' if own_pct >= 0 else ''}{own_pct:.1f}%" if has_history else ''

        # Determine flag and reason
        flag_val = ''
        matched_id = ''
        reason_text = ''
        retrieved_ids = []

        if is_candidate:
            # Retrieve and decide
            candidates = retriever.retrieve(route, week_monday)
            flag_val, matched_id, retrieved_ids = decision.decide(
                route, week_monday, candidates, is_valid_fn,
            )

            # Get matched note for reason
            matched_note = None
            if matched_id:
                matched_note = next(
                    (n for n in parsed_notes if n.note_id == matched_id), None
                )

            # Find nearby rejected note for context in Yes rows
            rejected_nearby = None
            if flag_val == 'Yes':
                rejected_nearby = reasons.find_nearby_rejected_note(
                    parsed_notes, route, week_monday,
                    matched_id, is_valid_fn,
                )

            reason_text = reasons.template_reason(
                flag_val, matched_note, route, week_monday,
                own_pct_str, candidates, rejected_nearby,
            )

            # Optional LLM reason rewriting
            if use_llm_reasons and llm_client is not None:
                reason_text = _rewrite_reason_with_llm(
                    llm_client, reason_text, route, week_monday,
                    own_pct_str, vs_peer, flag_val, matched_id,
                    matched_note, all_note_ids, refresh_llm,
                )

        # Build row dicts
        row_dict = {
            'route': route,
            'week_of': week_monday.isoformat() if isinstance(week_monday, date) else str(week_monday)[:10],
            'cost_per_tonne_km': cost,
            'vs_own_history': vs_own,
            'vs_similar_routes': vs_peer,
            'flagged': flag_val,
            'matched_note_id': matched_id,
            'reason': reason_text,
            # Extra fields for diagnostics
            'route_type': row['route_type'],
            'history_weeks_used': row['history_weeks_used'],
            'peer_count': row['peer_count'],
            'own_pct': own_pct,
            'peer_pct': peer_pct,
            'is_candidate': is_candidate,
            'retrieved_note_ids': ';'.join(retrieved_ids),
        }

        diagnostics_rows.append(row_dict)

        if is_candidate:
            output_rows.append(row_dict)
        elif all_rows:
            row_dict['flagged'] = 'No'
            output_rows.append(row_dict)

    # ── Phase 7: Write outputs ──
    logger.info('Phase 7: Writing output files...')
    # Sort output rows by route, then week_of for determinism
    output_rows.sort(key=lambda r: (r['route'], r['week_of']))
    diagnostics_rows.sort(key=lambda r: (r['route'], r['week_of']))

    output.write_output_csv(output_rows, output_csv, all_rows=all_rows)
    output.write_diagnostics_csv(diagnostics_rows, diagnostics_csv)

    # ── Summary ──
    n_candidates = sum(1 for r in output_rows if r['flagged'] in ('Yes', 'No (justified)'))
    n_yes = sum(1 for r in output_rows if r['flagged'] == 'Yes')
    n_justified = sum(1 for r in output_rows if r['flagged'] == 'No (justified)')

    summary = {
        'total_route_weeks': len(diagnostics_rows),
        'candidates': n_candidates,
        'flagged_yes': n_yes,
        'flagged_justified': n_justified,
        'output_rows': len(output_rows),
        'llm_calls': llm_client.call_count if llm_client else 0,
    }

    logger.info(
        '=== Pipeline complete: %d route-weeks, %d candidates '
        '(%d Yes, %d Justified), %d LLM calls ===',
        summary['total_route_weeks'], summary['candidates'],
        summary['flagged_yes'], summary['flagged_justified'],
        summary['llm_calls'],
    )

    return summary


def _rewrite_reason_with_llm(
    llm_client: BaseLLMClient,
    template_reason: str,
    route: str,
    week_monday: date,
    own_pct_str: str,
    vs_peer: str,
    flagged: str,
    matched_note_id: str,
    matched_note: ParsedNote | None,
    all_note_ids: list[str],
    refresh: bool,
) -> str:
    """Optionally rewrite the reason using the LLM.

    Falls back to template if validation fails.
    """
    facts = (
        f'Route: {route}\\n'
        f'Week: {week_monday.isoformat()}\\n'
        f'Own history change: {own_pct_str}\\n'
        f'Peer comparison: {vs_peer}\\n'
        f'Verdict: {flagged}\\n'
    )
    if matched_note and matched_note_id:
        facts += (
            f'Matched note: {matched_note_id} ({matched_note.start_date.isoformat()})\\n'
            f'Note text: {matched_note.text}\\n'
        )

    prompt = (
        f'Rewrite this freight cost analysis reason in exactly two plain '
        f'sentences. Use only these facts; do not add any cause, date or '
        f'number not listed here.\\n\\nFacts:\\n{facts}\\n'
        f'Current reason: {template_reason}'
    )

    try:
        response = llm_client.complete(
            prompt, purpose='reason_rewrite', max_tokens=200,
            bypass_cache=refresh,
        )
        llm_text = response.text.strip()

        # Validate
        matched_start = matched_note.start_date if matched_note else None
        if reasons.validate_llm_reason(
            llm_text, own_pct_str, flagged,
            matched_note_id, matched_start, all_note_ids,
        ):
            return llm_text
        else:
            logger.warning(
                'LLM reason failed validation for %s/%s; using template',
                route, week_monday,
            )
    except Exception as e:
        logger.error('LLM reason rewrite failed for %s/%s: %s', route, week_monday, e)

    return template_reason
