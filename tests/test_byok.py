"""Offline tests for byok.py - the Bring-Your-Own-Key resolution + redaction.
Pure functions, no Streamlit, no API, no cost. Run free, any time:

    /usr/bin/python3 tests/test_byok.py

These lock the security-critical promises: env-first resolution, a key is never
treated as present when blank, the format check rejects obvious junk before it
burns an API call, and ANYTHING key-shaped is masked before it can reach a log
or the UI (defence-in-depth against provider errors that echo a partial key).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from byok import key_from_env, resolve_key, validate_key_format, redact

CASES = []
def case(name):
    def wrap(fn):
        CASES.append((name, fn))
        return fn
    return wrap


# ---- key_from_env: read-only, blank normalised to None ----

@case("key_from_env: present value is returned")
def _():
    assert key_from_env({"OPENAI_API_KEY": "sk-abc123def"}) == "sk-abc123def"

@case("key_from_env: missing var -> None")
def _():
    assert key_from_env({}) is None

@case("key_from_env: blank string -> None (present-but-empty counts as missing)")
def _():
    assert key_from_env({"OPENAI_API_KEY": ""}) is None

@case("key_from_env: never mutates the environment mapping it is handed")
def _():
    env = {"OPENAI_API_KEY": "sk-keep"}
    key_from_env(env)
    assert env == {"OPENAI_API_KEY": "sk-keep"}, env


# ---- resolve_key: env-first, then pasted (BYOK), else none ----

@case("resolve_key: env present -> (env_key, 'env')")
def _():
    assert resolve_key("sk-env", "sk-paste") == ("sk-env", "env")   # env wins

@case("resolve_key: no env, session present -> (session_key, 'byok')")
def _():
    assert resolve_key(None, "sk-paste") == ("sk-paste", "byok")

@case("resolve_key: neither present -> (None, 'none')")
def _():
    assert resolve_key(None, None) == (None, "none")

@case("resolve_key: blank env falls through to session")
def _():
    assert resolve_key("", "sk-paste") == ("sk-paste", "byok")


# ---- validate_key_format: light sanity check, never echoes the key ----

@case("validate_key_format: None / blank -> (False, 'empty')")
def _():
    assert validate_key_format(None) == (False, "empty")
    assert validate_key_format("   ") == (False, "empty")

@case("validate_key_format: too short -> rejected")
def _():
    ok, msg = validate_key_format("sk-abc")
    assert ok is False and "short" in msg, (ok, msg)

@case("validate_key_format: missing sk- prefix -> rejected")
def _():
    ok, msg = validate_key_format("abcdefghij")
    assert ok is False and "sk-" in msg, (ok, msg)

@case("validate_key_format: well-formed key -> (True, 'ok')")
def _():
    assert validate_key_format("sk-abcdefgh") == (True, "ok")
    assert validate_key_format("sk-proj-abcdef123456") == (True, "ok")

@case("validate_key_format: surrounding whitespace is tolerated")
def _():
    assert validate_key_format("  sk-abcdefgh  ") == (True, "ok")

@case("validate_key_format: the message never contains the key itself")
def _():
    secret = "sk-supersecretvalue12345"
    ok, msg = validate_key_format(secret)
    assert ok is True and secret not in msg, msg


# ---- redact: mask anything key-shaped, leave the rest untouched ----

@case("redact: an sk- key inside an error string is masked")
def _():
    out = redact("Auth failed for key sk-abcdef123456 — retry")
    assert "sk-abcdef123456" not in out and "«redacted-key»" in out, out

@case("redact: sk-proj- keys are masked too")
def _():
    out = redact("using sk-proj-abc123def456ghi")
    assert "sk-proj-abc123def456ghi" not in out and "«redacted-key»" in out, out

@case("redact: two keys in one string are both masked")
def _():
    out = redact("sk-aaaaaa1111 and sk-bbbbbb2222")
    assert out.count("«redacted-key»") == 2 and "sk-" not in out, out

@case("redact: text with nothing key-shaped is returned unchanged")
def _():
    assert redact("a normal log line, no secrets") == "a normal log line, no secrets"

@case("redact: a too-short sk- fragment (under the min run) is left alone")
def _():
    assert redact("sk-abc") == "sk-abc"          # 3 chars after sk-, below the 6 floor

@case("redact: None / empty pass through")
def _():
    assert redact(None) is None
    assert redact("") == ""

@case("redact: coerces a non-string (e.g. an exception) and still masks")
def _():
    out = redact(RuntimeError("bad key sk-abcdef123456"))
    assert "sk-abcdef123456" not in out and "«redacted-key»" in out, out


def main():
    passed = 0
    for name, fn in CASES:
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL  {name}\n      {e}")
        except Exception as e:                          # noqa: BLE001
            print(f"ERROR {name}\n      {type(e).__name__}: {e}")
        else:
            print(f"pass  {name}")
            passed += 1
    print(f"\n{passed}/{len(CASES)} passed")
    sys.exit(0 if passed == len(CASES) else 1)


if __name__ == "__main__":
    main()
