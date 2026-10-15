#!/usr/bin/env python
"""Generate token and cost report from a cold run.

Runs one full cold run (with --refresh-llm) and writes logs/token_cost_report.md
summarizing:
- Number of LLM calls
- Total input tokens
- Total output tokens
- Estimated cost based on rates in .env / published rates
- Breakdown per call purpose
- Local embedding cost ($0.00)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
import config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate LLM token and cost report.")
    parser.add_argument("--stub-llm", action="store_true", help="Run with stub LLM client")
    parser.add_argument("--skip-run", action="store_true", help="Skip running pipeline; parse existing log")
    args = parser.parse_args()

    logs_dir = _PROJECT_ROOT / "logs"
    usage_log = config.LLM_USAGE_LOG
    report_file = config.TOKEN_COST_REPORT

    if not args.skip_run:
        # Clear existing usage log for an accurate cold run measurement
        if usage_log.exists():
            usage_log.unlink()

        # Run cold pipeline
        cmd = [sys.executable, str(_PROJECT_ROOT / "scripts" / "run.py"), "--refresh-llm"]
        if args.stub_llm:
            cmd.append("--stub-llm")

        print("Executing cold run with --refresh-llm...")
        res = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print(f"Error during run:\n{res.stderr}")
            sys.exit(res.returncode)

    # Parse logs/llm_usage.jsonl
    entries = []
    if usage_log.exists():
        with open(usage_log, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    total_calls = len(entries)
    uncached_calls = sum(1 for e in entries if not e.get("cached", False))
    cached_calls = sum(1 for e in entries if e.get("cached", False))

    total_in = sum(e.get("input_tokens", 0) for e in entries)
    total_out = sum(e.get("output_tokens", 0) for e in entries)

    # Cost calculation: published pricing vs actual .env pricing
    price_in = config.PRICE_IN
    price_out = config.PRICE_OUT
    cost_actual = (total_in / 1e6 * price_in) + (total_out / 1e6 * price_out)

    # Published reference pricing for Gemini 2.0 Flash: $0.10/M in, $0.40/M out
    ref_price_in = 0.10
    ref_price_out = 0.40
    cost_published = (total_in / 1e6 * ref_price_in) + (total_out / 1e6 * ref_price_out)

    # Breakdown by purpose
    by_purpose = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0})
    for e in entries:
        p = e.get("purpose", "unknown")
        by_purpose[p]["calls"] += 1
        by_purpose[p]["in"] += e.get("input_tokens", 0)
        by_purpose[p]["out"] += e.get("output_tokens", 0)

    # Build markdown report
    md = []
    md.append("# Token and Cost Report (Cold Run)")
    md.append("")
    md.append("This report measures the token consumption and cost of running the Freight Cost Watcher pipeline cold (empty cache / `--refresh-llm`).")
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append(f"- **Provider**: `{config.LLM_PROVIDER}`")
    md.append(f"- **Model**: `{config.LLM_MODEL}`")
    md.append(f"- **Total LLM Calls**: {total_calls} ({uncached_calls} uncached, {cached_calls} cached)")
    md.append(f"- **Total Input Tokens**: {total_in:,}")
    md.append(f"- **Total Output Tokens**: {total_out:,}")
    md.append(f"- **Total Tokens**: {total_in + total_out:,}")
    md.append(f"- **Estimated Run Cost**: **${cost_actual:.6f} USD** (based on configured rates)")
    md.append(f"- **Published List Price Rate**: ${ref_price_in}/M input, ${ref_price_out}/M output (Est: **${cost_published:.6f} USD**)")
    md.append("- **Vector Embedding Cost**: **$0.00** (Local CPU `all-MiniLM-L6-v2` / ChromaDB)")
    md.append("")
    md.append("## Breakdown by Purpose")
    md.append("")
    md.append("| Purpose | Calls | Input Tokens | Output Tokens | Subtotal Cost (Ref) |")
    md.append("|---|---|---|---|---|")
    for purpose, stats in sorted(by_purpose.items()):
        sub_cost = (stats["in"] / 1e6 * ref_price_in) + (stats["out"] / 1e6 * ref_price_out)
        md.append(f"| `{purpose}` | {stats['calls']} | {stats['in']:,} | {stats['out']:,} | ${sub_cost:.6f} |")
    md.append("")
    md.append("## Key Observations & Design Efficiency")
    md.append("")
    md.append("1. **Cache Efficiency**: In subsequent runs without `--refresh-llm`, note classification calls are cached on disk, reducing ongoing LLM calls to **0**.")
    md.append("2. **Zero Chat Overhead**: Chat / RAG generation is intentionally out of scope, eliminating conversational token bloat.")
    md.append("3. **Budget Guard**: The pipeline enforces `MAX_LLM_CALLS = 40` as a hard circuit breaker to prevent unexpected API costs.")
    md.append("4. **Deterministic Separation**: Numbers, route baselines, and note candidate retrieval never use LLM tokens.")

    report_content = "\n".join(md)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(report_content, encoding="utf-8")
    print(f"\nReport written to: {report_file}")
    print(report_content)


if __name__ == "__main__":
    main()
