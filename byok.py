"""
byok.py — Bring-Your-Own-Key resolution for the Live Review tab.

Pure functions — no Streamlit, no file I/O, and (critically) NO writes to
os.environ. The dashboard passes in the current environment and the key a
visitor pasted this session; these functions decide which key to use and
whether a live run can proceed.

This project talks to one provider (OpenAI), so there's a single key rather
than the three llm-eval-lab's byok.py resolves, but the security model is
identical.

Security model
--------------
  - The key is only ever READ — from the environment (a local `.env`) or from
    the pasted session value. It is NEVER written back to os.environ, a file, a
    cache, or a log. os.environ is process-global and shared across every
    visitor on a server instance, so writing a pasted key there could leak one
    session's key into another's. We never do that — the key flows as an
    explicit argument down to the SDK client (see llm._client).
  - On the deployed app there is no `.env`, so resolution falls through to the
    pasted session key, scoped to that one browser session's RAM.
"""

from __future__ import annotations

import re

ENV_VAR = "OPENAI_API_KEY"


def key_from_env(environ) -> str | None:
    """
    Pull the OpenAI key from an environment mapping (e.g. os.environ).

    Reads only — never mutates `environ`. An empty-string value is normalised
    to None so "present but blank" counts as missing.
    """
    return environ.get(ENV_VAR) or None


def resolve_key(env_key: str | None, session_key: str | None) -> tuple[str | None, str]:
    """
    Decide which key Live Review should use: env-first, then pasted (BYOK).

    Parameters
    ----------
    env_key     : key from the environment (a local `.env`), or None.
    session_key : key pasted this session (BYOK), or None.

    Returns
    -------
    (key, source)
      key    : the key to forward to the agent (None if neither is present;
               the dashboard then keeps Run disabled, so we never call OpenAI
               with no key).
      source : "env"  if the environment supplied it (no paste needed)
               "byok" if it came from the pasted session value
               "none" if there is no key from either source
    """
    if env_key:
        return env_key, "env"
    if session_key:
        return session_key, "byok"
    return None, "none"


def validate_key_format(key: str | None) -> tuple[bool, str]:
    """
    Light, non-cryptographic sanity check so obvious junk fails fast with a
    clean message instead of burning an API call. NEVER logs or echoes the key.

    Returns (ok, message). `ok=False` means the string is clearly not a key; it
    does NOT prove a key is valid — only OpenAI can do that.
    """
    if not key or not key.strip():
        return False, "empty"
    key = key.strip()
    if len(key) < 8:
        return False, "too short to be a real key"
    if not key.startswith("sk-"):
        return False, "OpenAI keys usually start with 'sk-'"
    return True, "ok"


# Mask anything sk-key-shaped. OpenAI keys (sk-… and sk-proj-…) all start with
# 'sk-', so one alternative covers them. Over-masking an unrelated 'sk-' string
# is harmless; under-masking a real key is not.
_KEY_RE = re.compile(r"(sk-[A-Za-z0-9_\-]{6,})")


def redact(text):
    """
    Mask anything that looks like an API key in a string before it is shown in
    the UI or written to a log. Defence-in-depth: provider auth errors sometimes
    echo a partial key, and we never want even that to surface. Returns the
    input unchanged when there is nothing key-shaped in it.
    """
    if not text:
        return text
    return _KEY_RE.sub("«redacted-key»", str(text))
