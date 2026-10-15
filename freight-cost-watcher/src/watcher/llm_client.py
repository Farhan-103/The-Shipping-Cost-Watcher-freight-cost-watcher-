"""LLM client with caching, stubbing, budget guard and usage logging.

Supports Google Gemini. Temperature 0, JSON mode where available.
Disk cache keyed by hash of provider, model, prompt version and prompt text.
"""
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    text: str
    input_tokens: int
    output_tokens: int
    cached: bool = False


class LLMBudgetExceeded(Exception):
    """Raised when the LLM call budget is exceeded."""


class BaseLLMClient:
    """Base class for LLM clients."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        cache_dir: Path,
        usage_log: Path,
        prompt_version: str = 'v1',
        max_calls: int = 40,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.usage_log = usage_log
        self.prompt_version = prompt_version
        self.max_calls = max_calls
        self._call_count = 0

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage_log.parent.mkdir(parents=True, exist_ok=True)

    @property
    def call_count(self) -> int:
        return self._call_count

    def _cache_key(self, prompt: str) -> str:
        """Compute a cache key from provider, model, prompt version and prompt."""
        raw = f'{self.provider}|{self.model}|{self.prompt_version}|{prompt}'
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f'{key}.json'

    def _load_cache(self, prompt: str) -> LLMResponse | None:
        """Load a cached response if it exists."""
        key = self._cache_key(prompt)
        path = self._cache_path(key)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                return LLMResponse(
                    text=data['text'],
                    input_tokens=data['input_tokens'],
                    output_tokens=data['output_tokens'],
                    cached=True,
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning('Cache read failed for %s: %s', key[:12], e)
        return None

    def _save_cache(self, prompt: str, response: LLMResponse) -> None:
        """Save a response to the disk cache."""
        key = self._cache_key(prompt)
        path = self._cache_path(key)
        data = {
            'text': response.text,
            'input_tokens': response.input_tokens,
            'output_tokens': response.output_tokens,
        }
        path.write_text(json.dumps(data, indent=2), encoding='utf-8')

    def _log_usage(self, purpose: str, response: LLMResponse) -> None:
        """Log usage to the JSONL usage log."""
        entry = {
            'purpose': purpose,
            'model': self.model,
            'provider': self.provider,
            'input_tokens': response.input_tokens,
            'output_tokens': response.output_tokens,
            'cached': response.cached,
        }
        with open(self.usage_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')

    def complete(
        self,
        prompt: str,
        purpose: str,
        max_tokens: int = 100,
        bypass_cache: bool = False,
    ) -> LLMResponse:
        """Send a prompt to the LLM and return the response.

        Checks cache first (unless bypass_cache). Enforces budget guard.
        """
        # Check cache
        if not bypass_cache:
            cached = self._load_cache(prompt)
            if cached is not None:
                logger.debug('Cache hit for %s', purpose)
                self._log_usage(purpose, cached)
                return cached

        # Budget guard
        if self._call_count >= self.max_calls:
            raise LLMBudgetExceeded(
                f'LLM call budget exceeded: {self._call_count}/{self.max_calls} calls used. '
                f'Increase MAX_LLM_CALLS in config to allow more calls.'
            )

        self._call_count += 1
        response = self._call_api(prompt, max_tokens)

        # Cache and log
        self._save_cache(prompt, response)
        self._log_usage(purpose, response)

        return response

    def _call_api(self, prompt: str, max_tokens: int) -> LLMResponse:
        """Override in subclasses to call the actual API."""
        raise NotImplementedError


class StubLLMClient(BaseLLMClient):
    """Stub LLM client for testing. Returns deterministic responses."""

    def __init__(self, **kwargs):
        # Don't require a real API key for stubs
        kwargs.setdefault('api_key', 'stub-key')
        super().__init__(**kwargs)
        self._stub_responses: dict[str, str] = {}

    def set_stub_response(self, prompt_contains: str, response: str) -> None:
        """Set a stub response for prompts containing the given text."""
        self._stub_responses[prompt_contains] = response

    def _call_api(self, prompt: str, max_tokens: int) -> LLMResponse:
        """Return a stub response based on the prompt content."""
        # Check for matching stub responses
        for key, response_text in self._stub_responses.items():
            if key in prompt:
                return LLMResponse(
                    text=response_text,
                    input_tokens=len(prompt) // 4,
                    output_tokens=len(response_text) // 4,
                )

        # Default: return a "not accepted" classification
        default = json.dumps({
            'states_cost_increase': False,
            'names_a_cause': False,
            'speculative_or_negated': False,
        })
        return LLMResponse(
            text=default,
            input_tokens=len(prompt) // 4,
            output_tokens=len(default) // 4,
        )


class GeminiLLMClient(BaseLLMClient):
    """Google Gemini LLM client using the google-genai SDK."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = None

    def _get_client(self):
        """Lazy-init the Gemini client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _call_api(self, prompt: str, max_tokens: int) -> LLMResponse:
        """Call the Gemini API with retries and backoff."""
        from google.genai import types

        client = self._get_client()

        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=max(max_tokens, 500),
            response_mime_type='application/json',
            thinking_config={'thinking_budget': 0},
        )

        last_error = None
        for attempt in range(5):
            try:
                if attempt > 0:
                    delay = 2 ** attempt
                    logger.info('Gemini retry %d after %ds delay', attempt, delay)
                    time.sleep(delay)

                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )

                text = response.text or ''
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

                # Small polite pacing delay between calls
                time.sleep(0.5)

                return LLMResponse(
                    text=text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            except Exception as e:
                last_error = e
                logger.warning('Gemini API error (attempt %d/5): %s', attempt + 1, e)

        raise RuntimeError(f'Gemini API failed after 5 attempts: {last_error}')


def create_llm_client(
    provider: str,
    model: str,
    api_key: str,
    cache_dir: Path,
    usage_log: Path,
    prompt_version: str = 'v1',
    max_calls: int = 40,
    stub: bool = False,
) -> BaseLLMClient:
    """Factory to create the appropriate LLM client."""
    kwargs = dict(
        provider=provider,
        model=model,
        api_key=api_key,
        cache_dir=cache_dir,
        usage_log=usage_log,
        prompt_version=prompt_version,
        max_calls=max_calls,
    )

    if stub:
        return StubLLMClient(**kwargs)

    if provider == 'google':
        return GeminiLLMClient(**kwargs)

    raise ValueError(f'Unsupported LLM provider: {provider}')
