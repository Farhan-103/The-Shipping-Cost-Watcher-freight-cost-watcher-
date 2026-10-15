"""Central configuration for Freight Cost Watcher."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / '.env')

# ── Paths ──
DATA_DIR = _PROJECT_ROOT / 'data'
OUTPUT_DIR = _PROJECT_ROOT / 'output'
LOGS_DIR = _PROJECT_ROOT / 'logs'
CACHE_DIR = _PROJECT_ROOT / 'cache'
EVAL_DIR = _PROJECT_ROOT / 'eval'

# ── Data files ──
SHIPMENT_CSV = DATA_DIR / 'shipment_records.csv'
NOTES_CSV = DATA_DIR / 'context_notes.csv'
SAMPLE_OUTPUT_CSV = DATA_DIR / 'sample_output_format_v2.csv'

# ── Output files ──
OUTPUT_CSV = OUTPUT_DIR / 'output.csv'
DIAGNOSTICS_CSV = OUTPUT_DIR / 'diagnostics_all_weeks.csv'

# ── Log files ──
LLM_USAGE_LOG = LOGS_DIR / 'llm_usage.jsonl'
RETRIEVAL_LOG = LOGS_DIR / 'retrieval_log.csv'
CLASSIFIER_DISAGREEMENTS_LOG = LOGS_DIR / 'classifier_disagreements.csv'
REPRO_REPORT = LOGS_DIR / 'repro_check.txt'
TOKEN_COST_REPORT = LOGS_DIR / 'token_cost_report.md'

# ── Flag thresholds ──
FLAG_OWN_PCT: float = 9.0
FLAG_PEER_PCT: float | None = None  # Off by default
HISTORY_WINDOW: int = 8  # Up to 8 prior weeks

# ── Route type mapping (from spec) ──
ROUTE_TYPE_MAP: dict[str, str] = {
    'Mumbai-Pune': 'Short',
    'Delhi-Jaipur': 'Short',
    'Ahmedabad-Mumbai': 'Medium',
    'Chennai-Bangalore': 'Medium',
    'Kolkata-Bhubaneswar': 'Medium',
    'Mumbai-Delhi': 'Long',
    'Delhi-Chennai': 'Long',
}

# ── LLM settings ──
LLM_PROVIDER: str = os.getenv('LLM_PROVIDER', 'google')
LLM_MODEL: str = os.getenv('LLM_MODEL', 'gemini-3.8-flash')
LLM_API_KEY: str = os.getenv('LLM_API_KEY', '')
LLM_SAMPLES_PER_NOTE: int = 1
MAX_LLM_CALLS: int = 40
LLM_MAX_CLASSIFICATION_TOKENS: int = 100
LLM_MAX_REWRITE_TOKENS: int = 200
PROMPT_VERSION: str = 'v1'

# ── Cost rates (from .env, for cost reporting) ──
PRICE_IN: float = float(os.getenv('PRICE_IN', '0.0'))
PRICE_OUT: float = float(os.getenv('PRICE_OUT', '0.0'))

# ── Embedding ──
EMBEDDING_MODEL: str = 'all-MiniLM-L6-v2'
CHROMA_COLLECTION: str = 'context_notes'
