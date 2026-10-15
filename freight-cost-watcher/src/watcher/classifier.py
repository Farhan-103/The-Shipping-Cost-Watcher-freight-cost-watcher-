"""Keyword-based and LLM-based classifier for context notes.

A note is a valid reason only if it says costs/rates/prices went up,
names a cause, and is not speculative or negated.

Two independent checks, both must pass:
1. keyword_says_cost_up() — deterministic keyword check with guards and cause validation.
2. llm_says_cost_up() — LLM check at temperature 0 returning JSON.

Combined decision: valid = keyword AND LLM both accept.
Disagreements are logged and treated as not valid.
"""
import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Keyword lists ──
UP_WORDS = [
    'higher', 'surcharge', 'pushing up', 'rose', 'increase', 'increasing',
    'went up', 'charged more', 'raised', 'pushed up', 'jumped',
    'rising', 'elevated', 'hiked', 'hike', 'inflated',
    'costlier', 'premium', 'additional charge',
]

NEGATIONS = [
    'no ', 'not ', 'without ', 'did not', 'remained',
    'no significant', 'not significantly', 'did not affect',
    'no rate change', 'no major', 'without a rate change',
]

HEDGES = [
    'may ', 'might ', 'could ', 'expected to', 'likely',
]

COST_DOWN_OR_NEUTRAL = [
    'waived', 'despite', 'flat', 'unchanged', 'stayed',
    'stable', 'absorbed', 'remained normal', 'normal operations',
    'returned to normal', 'improved', 'completed', 'back to normal',
    'pre-storm levels', 'last year',
]

CAUSE_WORDS = [
    'due to', 'because', 'flood', 'festival', 'diesel', 'fuel',
    'surcharge', 'strike', 'closure', 'detour', 'delay', 'mandate',
    'toll', 'landslide', 'rain', 'weather', 'storm', 'congestion',
    'levy', 'subsidy', 'overtime', 'maintenance', 'fog',
]


def keyword_says_cost_up(text: str) -> bool:
    """Check if the note text indicates costs went up, using keywords and guards.

    Returns True only if:
    - At least one UP_WORD is present, AND
    - No NEGATION, HEDGE, or COST_DOWN_OR_NEUTRAL word is present, AND
    - At least one concrete CAUSE_WORD is present (guarding against "costs were higher" with no cause).
    """
    t = text.lower()
    has_up = any(w in t for w in UP_WORDS)
    has_negation = any(w in t for w in NEGATIONS)
    has_hedge = any(w in t for w in HEDGES)
    has_neutral = any(w in t for w in COST_DOWN_OR_NEUTRAL)
    has_cause = any(w in t for w in CAUSE_WORDS)

    risky = has_negation or has_hedge or has_neutral
    return has_up and not risky and has_cause


# ── LLM classification prompt ──
CLASSIFICATION_PROMPT_TEMPLATE = """You are a freight cost analyst. Read the following context note and answer three questions about it. Return ONLY valid JSON with exactly these three boolean fields.

Context note:
\"{note_text}\"

Questions:
1. states_cost_increase: Does this note explicitly state that freight costs, rates, or prices increased? Answer true ONLY if the note clearly says costs/rates/prices went up. Answer false if unsure or if the note says costs were not affected, remained stable, or were absorbed.
2. names_a_cause: Does the note name a specific cause for the cost increase (e.g., flooding, festival, diesel price rise, strike)? Answer true only if a concrete cause is stated.
3. speculative_or_negated: Is the cost increase statement speculative (using words like "may", "might", "could", "expected to") or negated (saying costs did NOT increase, were waived, remained flat, etc.)? Answer true if speculative or negated.

Return ONLY this JSON (no markdown, no explanation):
{{"states_cost_increase": <bool>, "names_a_cause": <bool>, "speculative_or_negated": <bool>}}"""


def build_classification_prompt(note_text: str) -> str:
    """Build the LLM classification prompt for a note."""
    return CLASSIFICATION_PROMPT_TEMPLATE.format(note_text=note_text)


def parse_llm_classification(response_text: str) -> dict | None:
    """Parse the LLM response JSON. Returns dict or None if invalid."""
    try:
        # Strip any markdown code fences
        text = response_text.strip()
        if text.startswith('```'):
            lines = text.split('\n')
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines if not l.strip().startswith('```')]
            text = '\n'.join(lines)

        data = json.loads(text)

        # Validate required fields
        required = {'states_cost_increase', 'names_a_cause', 'speculative_or_negated'}
        if not required.issubset(data.keys()):
            logger.warning('LLM response missing fields: %s', required - data.keys())
            return None

        # Validate all are booleans
        for key in required:
            if not isinstance(data[key], bool):
                logger.warning('LLM response field %s is not bool: %s', key, data[key])
                return None

        return data

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning('Failed to parse LLM classification response: %s', e)
        return None


def llm_accepts(parsed: dict) -> bool:
    """Check if the LLM classification accepts the note as a valid reason.

    Accepts when: states_cost_increase=True, names_a_cause=True,
    speculative_or_negated=False.
    """
    return (
        parsed['states_cost_increase'] is True
        and parsed['names_a_cause'] is True
        and parsed['speculative_or_negated'] is False
    )


def is_valid_reason(
    note_text: str,
    llm_result: dict | None = None,
    keyword_only: bool = False,
) -> bool:
    """Combined decision: both keyword AND LLM checks must accept.

    Args:
        note_text: The note text to classify.
        llm_result: Pre-computed LLM classification dict, or None for keyword-only.
        keyword_only: If True, skip LLM check (used before LLM is integrated).

    Returns:
        True if the note is a valid reason for a cost increase.
    """
    kw = keyword_says_cost_up(note_text)

    if keyword_only or llm_result is None:
        return kw

    llm_ok = llm_accepts(llm_result)
    return kw and llm_ok


def log_disagreement(
    log_path: Path,
    note_id: str,
    keyword_result: bool,
    llm_result: bool,
    note_text: str,
) -> None:
    """Log a keyword/LLM disagreement to the disagreements CSV."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()

    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['note_id', 'keyword_result', 'llm_result', 'note_text'])
        writer.writerow([note_id, keyword_result, llm_result, note_text[:200]])
