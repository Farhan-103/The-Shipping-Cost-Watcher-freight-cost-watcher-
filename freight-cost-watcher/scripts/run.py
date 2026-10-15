#!/usr/bin/env python
"""CLI entry point for the Freight Cost Watcher pipeline."""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so config and src can be imported
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402
from src.watcher.pipeline import run_pipeline  # noqa: E402
from src.watcher.llm_client import create_llm_client  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description='Freight Cost Watcher: detect and explain unusual shipping cost rises.',
    )
    parser.add_argument(
        '--all-rows', action='store_true',
        help='Include all 728 route-weeks in output (non-candidates flagged as No).',
    )
    parser.add_argument(
        '--no-rag', action='store_true',
        help='Use filter-only retrieval (no ChromaDB/embedding).',
    )
    parser.add_argument(
        '--no-llm', action='store_true',
        help='Skip LLM classification; use keyword-only classifier.',
    )
    parser.add_argument(
        '--llm-reasons', action='store_true',
        help='Use LLM to rewrite reason text (off by default).',
    )
    parser.add_argument(
        '--refresh-llm', action='store_true',
        help='Bypass LLM cache and make fresh API calls.',
    )
    parser.add_argument(
        '--stub-llm', action='store_true',
        help='Use stub LLM client (for testing without API).',
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable debug logging.',
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )

    # Create LLM client
    use_llm = not args.no_llm
    llm_client = None

    if use_llm:
        if args.stub_llm or not config.LLM_API_KEY:
            if not config.LLM_API_KEY:
                logging.warning(
                    'No LLM_API_KEY set in .env; running in stub mode. '
                    'Set LLM_PROVIDER, LLM_MODEL, and LLM_API_KEY in .env '
                    'to use a real LLM.'
                )
            llm_client = create_llm_client(
                provider=config.LLM_PROVIDER,
                model=config.LLM_MODEL,
                api_key=config.LLM_API_KEY or 'stub',
                cache_dir=config.CACHE_DIR,
                usage_log=config.LLM_USAGE_LOG,
                prompt_version=config.PROMPT_VERSION,
                max_calls=config.MAX_LLM_CALLS,
                stub=True,
            )
        else:
            llm_client = create_llm_client(
                provider=config.LLM_PROVIDER,
                model=config.LLM_MODEL,
                api_key=config.LLM_API_KEY,
                cache_dir=config.CACHE_DIR,
                usage_log=config.LLM_USAGE_LOG,
                prompt_version=config.PROMPT_VERSION,
                max_calls=config.MAX_LLM_CALLS,
                stub=False,
            )

    # Run the pipeline
    summary = run_pipeline(
        shipment_csv=config.SHIPMENT_CSV,
        notes_csv=config.NOTES_CSV,
        output_csv=config.OUTPUT_CSV,
        diagnostics_csv=config.DIAGNOSTICS_CSV,
        retrieval_log=config.RETRIEVAL_LOG,
        disagreements_log=config.CLASSIFIER_DISAGREEMENTS_LOG,
        flag_own_pct=config.FLAG_OWN_PCT,
        flag_peer_pct=config.FLAG_PEER_PCT,
        history_window=config.HISTORY_WINDOW,
        use_rag=not args.no_rag,
        use_llm_classifier=use_llm,
        use_llm_reasons=args.llm_reasons,
        all_rows=args.all_rows,
        llm_client=llm_client,
        refresh_llm=args.refresh_llm,
    )

    # Print summary
    print(f"\n{'='*60}")
    print('Freight Cost Watcher — Run Summary')
    print(f"{'='*60}")
    print(f"Total route-weeks:   {summary['total_route_weeks']}")
    print(f"Candidates:          {summary['candidates']}")
    print(f"  Flagged (Yes):     {summary['flagged_yes']}")
    print(f"  Justified:         {summary['flagged_justified']}")
    print(f"Output rows:         {summary['output_rows']}")
    print(f"LLM calls:           {summary['llm_calls']}")
    print(f"Output:              {config.OUTPUT_CSV}")
    print(f"Diagnostics:         {config.DIAGNOSTICS_CSV}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
