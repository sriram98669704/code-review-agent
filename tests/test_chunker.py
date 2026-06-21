"""Offline tests for chunker.py - the deterministic AST walk that GUARANTEES every
file and every top-level thing gets seen. Pure parsing, no API, no cost:

    /usr/bin/python3 tests/test_chunker.py

These lock the coverage promise: functions/methods come back as 'code' chunks
(they later get security + duplicate checks), everything else (imports, top-level
assignments like a hardcoded API_KEY) comes back as 'module' chunks (security
only), nested defs don't double-count, ignored dirs (venv, __pycache__) are
skipped, and the walk is sorted + reproducible. Synthetic temp files, so the
test is self-contained and never depends on the sample repo's exact contents.
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chunker import walk_files, chunk_file, scan_file, IGNORE

CASES = []
def case(name):
    def wrap(fn):
        CASES.append((name, fn))
        return fn
    return wrap


def write(dirpath, name, text):
    p = Path(dirpath) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


MIXED = '''\
import os
from sys import argv

API_KEY = "sk-hardcoded"


def top_level(x):
    def inner(y):          # a nested def is NOT its own top-level chunk
        return y + 1
    return inner(x)


async def fetch(url):
    return url


class Service:
    def handle(self, req):
        return req

    async def aclose(self):
        return None
'''


@case("chunk_file: every top-level item becomes exactly one chunk, in order")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = write(d, "m.py", MIXED)
        chunks = chunk_file(p)
        names = [(c["name"], c["kind"]) for c in chunks]
    assert names == [
        ("<Import>", "module"),
        ("<ImportFrom>", "module"),
        ("<Assign>", "module"),
        ("top_level", "code"),
        ("fetch", "code"),
        ("Service.handle", "code"),
        ("Service.aclose", "code"),
    ], names

@case("chunk_file: a nested def stays inside its parent, not its own chunk")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = write(d, "m.py", MIXED)
        names = [c["name"] for c in chunk_file(p)]
    assert "inner" not in names, names          # the inner def is part of top_level's code

@case("chunk_file: a 'code' chunk's source slice is the real function text")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = write(d, "m.py", MIXED)
        fetch = next(c for c in chunk_file(p) if c["name"] == "fetch")
    assert fetch["code"].startswith("async def fetch(url):"), fetch["code"]
    assert "return url" in fetch["code"], fetch["code"]
    assert fetch["start"] <= fetch["end"], fetch

@case("chunk_file: line numbers are 1-based and bracket the function")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = write(d, "m.py", "x = 1\n\n\ndef f():\n    return 2\n")
        f = next(c for c in chunk_file(p) if c["name"] == "f")
    assert f["start"] == 4 and f["end"] == 5, f

@case("chunk_file: a module docstring is one 'module' chunk (an Expr), not skipped")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = write(d, "m.py", '"""just a docstring"""\n')
        chunks = chunk_file(p)
    assert len(chunks) == 1 and chunks[0]["kind"] == "module", chunks

@case("chunk_file: an empty file yields zero chunks (nothing to walk)")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = write(d, "empty.py", "")
        assert chunk_file(p) == []

@case("chunk_file: a class with no methods still yields the class as one module chunk")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = write(d, "m.py", "class Empty:\n    pass\n")
        chunks = chunk_file(p)
    # ClassDef with no FunctionDef bodies contributes no 'code' chunks
    assert all(c["kind"] != "code" for c in chunks), chunks
    assert chunks == [], chunks                 # the for-loop over an all-pass body adds nothing


# ---- walk_files: sorted, .py only, ignored dirs skipped ----

@case("walk_files: returns every .py, sorted, and nothing else")
def _():
    with tempfile.TemporaryDirectory() as d:
        write(d, "b.py", "")
        write(d, "a.py", "")
        write(d, "notes.txt", "not python")
        write(d, "pkg/c.py", "")
        files = [p.name for p in walk_files(d)]
    assert files == ["a.py", "b.py", "c.py"], files   # sorted, .txt excluded

@case("walk_files: ignored dirs (venv, __pycache__, .git, _archive) are skipped")
def _():
    with tempfile.TemporaryDirectory() as d:
        write(d, "keep.py", "")
        for ig in sorted(IGNORE):
            write(d, f"{ig}/buried.py", "")
        files = [p.name for p in walk_files(d)]
    assert files == ["keep.py"], files

@case("walk_files: nested package dirs are walked recursively")
def _():
    with tempfile.TemporaryDirectory() as d:
        write(d, "deep/a/b/c.py", "")
        files = walk_files(d)
    assert len(files) == 1 and files[0].name == "c.py", files


# ---- scan_file: one read + one parse shared by chunks AND resolver bindings ----

@case("scan_file: returns (chunks, bindings) and the chunks match chunk_file's")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = write(d, "m.py", MIXED)
        chunks, bindings = scan_file(p)
        direct = chunk_file(p)
    assert [c["name"] for c in chunks] == [c["name"] for c in direct], chunks
    assert bindings is not None                 # resolver.file_bindings ran on the same parse


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
