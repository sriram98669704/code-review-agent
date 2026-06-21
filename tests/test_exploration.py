"""Offline tests for callgraph.summarize_exploration - the data behind the dashboard's
'agentic exploration' block. Pure AST, no API, no cost. Run free, any time:

    /usr/bin/python3 tests/test_exploration.py

summarize_exploration takes the helpers the agent opened on its OWN (read_function)
and, against the SAME bounds triage walks, sorts each into: a helper the guaranteed
pass also reaches, or one BEYOND it - and for the beyond ones, WHY (no-seed / depth /
width / off-path / unresolvable / not-on-chain), read straight off program facts. The
two headline cases are locked against the real shipped sample, so the block can never
claim the agent went somewhere the guaranteed pass actually reaches (or vice-versa).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from test_callgraph import _real_sample            # builds the real sample's index + facts
from callgraph import summarize_exploration


CASES = []
def case(name):
    def wrap(fn):
        CASES.append((name, fn))
        return fn
    return wrap


def read(*names):
    """Fake read_function results for `names`, resolved against the real sample so each
    carries the right 'path' (a name in two files yields one def per file)."""
    idx, _ = _real_sample()
    out = []
    for n in names:
        defs = idx.get(n, [])
        out.append({"name": n,
                    "definitions": [{"path": d["path"], "code": d["code"]} for d in defs]})
    return out


def flag(*pairs):
    """Flagged-function dicts {name, path} - the entries whose chains triage walks."""
    return [{"name": n, "path": p} for n, p in pairs]


def summarize(reads, flagged):
    idx, rf = _real_sample()
    return summarize_exploration(reads, flagged, idx, rf)


def only(out):
    assert len(out["opened"]) == 1, out["opened"]
    return out["opened"][0]


# ---- beyond the guaranteed pass: the two reasons the block must name correctly ----

@case("no-seed: agent opens guard_current for serve_current (global, no taint seed)")
def _():
    out = summarize(read("guard_current"), flag(("serve_current", "api.py")))
    o = only(out)
    assert o["beyond_forced"] and o["reason"] == "no-seed", o
    assert o["entry"] == "serve_current" and o["helper"] == "guard_current", o
    assert out["beyond_count"] == 1 and out["forced_count"] == 0, out


@case("depth: agent opens seal_archive for fetch_archive (fix sits 4 hops down)")
def _():
    out = summarize(read("seal_archive"), flag(("fetch_archive", "downloads.py")))
    o = only(out)
    assert o["beyond_forced"] and o["reason"] == "depth", o
    assert o["entry"] == "fetch_archive" and o["helper_file"] == "db.py", o


# ---- helpers the guaranteed pass ALSO walks: shown, attributed, but not starred ----

@case("forced (same file, 1 hop): safe_name is on read_export's guaranteed walk")
def _():
    out = summarize(read("safe_name"), flag(("read_export", "api.py")))
    o = only(out)
    assert not o["beyond_forced"] and o["reason"] is None, o
    assert o["entry"] == "read_export", o


@case("forced (cross-file, 2 hops): strip_traversal is on fetch_report's walk")
def _():
    out = summarize(read("strip_traversal"), flag(("fetch_report", "downloads.py")))
    o = only(out)
    assert not o["beyond_forced"] and o["entry"] == "fetch_report", o


# ---- both buckets together: overlap first, the ⭐ beyond ones last ----

@case("mixed run: one forced helper and one beyond-forced helper, sorted")
def _():
    out = summarize(read("safe_name", "guard_current"),
                    flag(("read_export", "api.py"), ("serve_current", "api.py")))
    assert out["forced_count"] == 1 and out["beyond_count"] == 1, out
    assert out["opened"][0]["beyond_forced"] is False, out["opened"]   # overlap first
    assert out["opened"][1]["beyond_forced"] is True, out["opened"]    # the ⭐ one last


# ---- graceful degradation: the agent opens something off any flagged chain ----

@case("not-on-chain: a helper no flagged function calls -> entry None, named honestly")
def _():
    out = summarize(read("user_exists"), flag(("read_export", "api.py")))
    o = only(out)
    assert o["beyond_forced"] and o["reason"] == "not-on-chain" and o["entry"] is None, o


# ---- robustness: error dicts and repeats never produce a phantom row ----

@case("a read_function error dict (phantom name) is skipped, not rendered")
def _():
    out = summarize([{"error": "function 'nope' not found"}], flag(("read_export", "api.py")))
    assert out["opened"] == [] and out["beyond_count"] == 0, out


@case("opening the same helper twice yields ONE row")
def _():
    out = summarize(read("safe_name") + read("safe_name"), flag(("read_export", "api.py")))
    assert len(out["opened"]) == 1, out["opened"]


@case("no reads at all -> empty summary (the empty-state block)")
def _():
    out = summarize([], flag(("read_export", "api.py")))
    assert out["opened"] == [] and out["forced_count"] == 0 and out["beyond_count"] == 0, out


def main():
    passed = 0
    for name, fn in CASES:
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL  {name}\n      {e}")
        except Exception as e:                          # noqa: BLE001 - surface any error clearly
            print(f"ERROR {name}\n      {type(e).__name__}: {e}")
        else:
            print(f"pass  {name}")
            passed += 1
    print(f"\n{passed}/{len(CASES)} passed")
    sys.exit(0 if passed == len(CASES) else 1)


if __name__ == "__main__":
    main()
