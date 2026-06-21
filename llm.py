"""Shared OpenAI access: one client, one place, with rate-limit retry baked in.

The policy check, the duplicate detector, and the agent loop all talk to OpenAI.
Keeping the client and the retry logic here means we don't repeat ourselves (our
own duplicate detector would flag two copies of embed!), and a single rate-limit
backoff protects EVERY call - chat and embeddings alike. It is also the one spot
that builds the client, so swapping in a user-supplied key later touches nothing
else.
"""

import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv(Path(__file__).parent / ".env")

MODEL = "gpt-4o"               # the judge + duplicate checks - deterministic (temperature=0, seed=0)
AGENT_MODEL = "gpt-4o"         # the agent's brain - same model as the judge for now (one-char flip to upgrade)
EMBED_MODEL = "text-embedding-3-small"

MAX_RETRIES = 5           # attempts before we give up on a rate-limited call
BACKOFF_BASE = 2          # seconds; the wait grows BACKOFF_BASE ** attempt

# A per-request network ceiling. WITHOUT this the OpenAI client falls back to the
# SDK default (~10 minutes), so one stalled connection makes the whole review look
# hung - it sits at 0% CPU waiting on a socket that will never answer. A tight
# timeout makes a stalled call fail fast; REQUEST_RETRIES then lets the SDK
# transparently re-issue a timed-out or dropped request before the error surfaces.
REQUEST_TIMEOUT = 60      # seconds per HTTP request - generous for a judge call, far below 10 min
REQUEST_RETRIES = 2       # SDK-level auto-retry on timeout / connection error

_default_client = None    # the CLI's env-based client, built lazily on first use


def _client(api_key=None):
    """Pick the OpenAI client for one call.

    With a pasted key (BYOK, from the UI) we build a client bound to THAT key and
    nothing else - the key flows in as an argument and is NEVER written to
    os.environ, so one session's key can't leak into another's. With no key (the
    CLI) we reuse a default client that read OPENAI_API_KEY from .env. We build
    that default lazily, so importing this module never needs a key - a deployed
    UI with no .env can still load and wait for the user to paste one.

    Every client carries REQUEST_TIMEOUT + REQUEST_RETRIES so no single call can
    hang the run on a stalled connection."""
    if api_key:
        return OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT, max_retries=REQUEST_RETRIES)
    global _default_client
    if _default_client is None:
        _default_client = OpenAI(timeout=REQUEST_TIMEOUT, max_retries=REQUEST_RETRIES)
    return _default_client


def _with_retry(call):
    """Run an OpenAI call, backing off and retrying if we hit the rate limit.

    Real repos have enough chunks to trip OpenAI's per-minute limit; the toy
    sample never did. Waits 1s, 2s, 4s, 8s between tries, then gives up."""
    for attempt in range(MAX_RETRIES):
        try:
            return call()
        except RateLimitError:
            if attempt == MAX_RETRIES - 1:    # out of attempts -> let it raise
                raise
            time.sleep(BACKOFF_BASE ** attempt)


def chat(messages, api_key=None, model=None, **kwargs):
    """A chat completion, with rate-limit retry. `model` overrides the default
    (MODEL): the agent loop passes AGENT_MODEL to drive its decisions, while the
    judge and duplicate checks keep the deterministic default."""
    return _with_retry(
        lambda: _client(api_key).chat.completions.create(model=model or MODEL, messages=messages, **kwargs))


def embed(text, api_key=None):
    """Embed one piece of text into a vector, with rate-limit retry."""
    resp = _with_retry(
        lambda: _client(api_key).embeddings.create(model=EMBED_MODEL, input=text))
    return resp.data[0].embedding


def embed_many(texts, api_key=None):
    """Embed a LIST of texts in one call; returns vectors in the same order."""
    resp = _with_retry(
        lambda: _client(api_key).embeddings.create(model=EMBED_MODEL, input=texts))
    return [d.embedding for d in resp.data]
