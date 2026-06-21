"""Offline tests for the DETERMINISTIC security checks in security.py - the rules
answered by reading the AST, not by asking the model. No API, no cost, but the
module imports chromadb, so run it with the project venv (not bare python3):

    venv/bin/python tests/test_security_ast.py

The one deterministic rule today is no-bare-except: a broad `except` whose body
never surfaces the error (no raise / log / print) silently swallows it. Because a
parser answers this with certainty, it never reaches the LLM - which is the whole
point (the model used to disagree with itself on two IDENTICAL except blocks).
These tests pin the AST logic: what counts as broad, what counts as surfacing,
that the whole handler body is searched, and that the reported line is correct.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from security import _check_bare_except, _surfaces_error

CASES = []
def case(name):
    def wrap(fn):
        CASES.append((name, fn))
        return fn
    return wrap


def chunk(code, start=1, name="f", path="m.py"):
    """A minimal chunk dict - exactly the keys _check_bare_except reads."""
    return {"code": code, "start": start, "name": name, "path": path}

def handler(code):
    """First ExceptHandler node in `code` - for testing _surfaces_error directly."""
    import ast
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            return node
    raise AssertionError("no ExceptHandler in code")


# ---- _check_bare_except: which broad handlers get flagged ----

@case("bare `except:` that only passes is flagged")
def _():
    f = _check_bare_except(chunk("try:\n    risky()\nexcept:\n    pass\n"))
    assert len(f) == 1 and f[0]["rule_id"] == "no-bare-except", f

@case("`except Exception:` that only passes is flagged")
def _():
    f = _check_bare_except(chunk("try:\n    risky()\nexcept Exception:\n    pass\n"))
    assert len(f) == 1, f

@case("`except BaseException:` that only passes is flagged")
def _():
    f = _check_bare_except(chunk("try:\n    risky()\nexcept BaseException:\n    pass\n"))
    assert len(f) == 1, f

@case("a broad except that RAISES is not flagged (re-surfaces the error)")
def _():
    f = _check_bare_except(chunk("try:\n    risky()\nexcept Exception:\n    raise\n"))
    assert f == [], f

@case("a broad except that LOGS is not flagged (logging.error surfaces it)")
def _():
    f = _check_bare_except(chunk(
        "try:\n    risky()\nexcept Exception as e:\n    logging.error(e)\n"))
    assert f == [], f

@case("a broad except that PRINTS is not flagged")
def _():
    f = _check_bare_except(chunk("try:\n    risky()\nexcept Exception:\n    print('boom')\n"))
    assert f == [], f

@case("a SPECIFIC exception type is always safe, even when it only passes")
def _():
    f = _check_bare_except(chunk("try:\n    risky()\nexcept ValueError:\n    pass\n"))
    assert f == [], f

@case("surfacing nested inside an if/with still counts as surfacing (whole body walked)")
def _():
    code = ("try:\n    risky()\nexcept Exception:\n    if cleanup_needed():\n"
            "        with lock:\n            raise\n")
    assert _check_bare_except(chunk(code)) == []

@case("a commented-out print does NOT count - the error still vanishes -> flagged")
def _():
    f = _check_bare_except(chunk("try:\n    risky()\nexcept Exception:\n    pass  # print(e)\n"))
    assert len(f) == 1, f

@case("two swallowing handlers yield two findings")
def _():
    code = ("try:\n    a()\nexcept ValueError:\n    pass\nexcept Exception:\n    pass\n"
            "try:\n    b()\nexcept:\n    pass\n")
    # ValueError is safe; the broad Exception and the bare except are each flagged
    assert len(_check_bare_except(chunk(code))) == 2, _check_bare_except(chunk(code))

@case("the reported line applies the chunk's start offset")
def _():
    # handler sits on relative line 3 of the snippet; with start=10 -> file line 12
    f = _check_bare_except(chunk("try:\n    risky()\nexcept:\n    pass\n", start=10))
    assert f[0]["line"] == 12, f

@case("the finding carries path, name, severity, and a non-empty fix")
def _():
    f = _check_bare_except(chunk("try:\n    x()\nexcept:\n    pass\n", name="handler", path="api.py"))
    assert f[0]["name"] == "handler" and f[0]["path"] == "api.py", f
    assert f[0]["severity"] and f[0]["fix"].strip(), f

@case("clean code with no try/except yields nothing")
def _():
    assert _check_bare_except(chunk("def f(x):\n    return x + 1\n")) == []


# ---- _surfaces_error: the predicate behind the above ----

@case("_surfaces_error: a raise surfaces")
def _():
    assert _surfaces_error(handler("try:\n    x()\nexcept Exception:\n    raise\n")) is True

@case("_surfaces_error: a logging call surfaces (attr name in the surface set)")
def _():
    assert _surfaces_error(handler("try:\n    x()\nexcept Exception:\n    logger.exception('e')\n")) is True

@case("_surfaces_error: a bare-name log() call surfaces")
def _():
    assert _surfaces_error(handler("try:\n    x()\nexcept Exception:\n    log('e')\n")) is True

@case("_surfaces_error: pass does NOT surface")
def _():
    assert _surfaces_error(handler("try:\n    x()\nexcept Exception:\n    pass\n")) is False

@case("_surfaces_error: an unrelated call (cleanup) does NOT surface")
def _():
    assert _surfaces_error(handler("try:\n    x()\nexcept Exception:\n    cleanup()\n")) is False


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
