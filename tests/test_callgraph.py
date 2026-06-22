"""Offline tests for callgraph.py - taint-guided interprocedural expansion. Pure AST,
no API, no cost. Run free, any time:

    /usr/bin/python3 tests/test_callgraph.py

Each test asserts WHICH helper sources triage would get to see when it follows the
call chain from a flagged function along the tainted data - the deep helpers it must
reach, and the non-tainted branches it must prune - plus the guards (depth, count,
cycles) that keep the walk finite. The final two tests lock the most important
property: on the shipped sample the bundle is byte-identical to the one-hop bundle
triage used before, so wiring this in does NOT change the sample's verdict.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from test_resolver import build                 # the same index+repo_facts builder
from callgraph import gather_evidence


CASES = []
def case(name):
    def wrap(fn):
        CASES.append((name, fn))
        return fn
    return wrap


def run(files, start, **kw):
    """gather_evidence from function `start`; return (helper bare names, sinks, full)."""
    idx, rf = build(files)
    ev = gather_evidence(idx[start][0], idx, rf, **kw)
    names = {k.split("::", 1)[1] for k in ev["helpers"]}
    return names, ev["sinks"], ev


# The exact example from the question: user_input reaches db.execute three hops down.
CANON = {"app.py": (
    "import db\n\n"
    "def api_handler(user_input):\n"
    "    query = build_query(user_input)\n"
    "    return run_query(query)\n\n"
    "def build_query(x):\n"
    "    return \"SELECT * FROM users WHERE id = \" + x\n\n"
    "def run_query(q):\n"
    "    db.execute(q)\n"
)}


@case("3-hop chain: taint reaches the sink past two helpers")
def _():
    names, sinks, _ = run(CANON, "api_handler")
    assert names == {"build_query", "run_query"}, names
    assert "db.execute" in sinks, sinks


@case("taint-guided: a call that receives no tainted value is pruned")
def _():
    files = {"app.py": (
        "def handler(user_input):\n"
        "    safe = build_query(user_input)\n"
        "    const = 'static'\n"
        "    log_page(const)\n"
        "    return run(safe)\n\n"
        "def build_query(x):\n    return x\n\n"
        "def log_page(p):\n    return p\n\n"
        "def run(q):\n    return q\n"
    )}
    names, _, _ = run(files, "handler")
    assert names == {"build_query", "run"}, names      # log_page pruned: const is not tainted


@case("the sanitiser on the path is included so the model can judge it")
def _():
    files = {"app.py": (
        "import db\n\n"
        "def handler(user_input):\n"
        "    return run_query(escape(user_input))\n\n"
        "def escape(s):\n    return s.replace(\"'\", \"''\")\n\n"
        "def run_query(q):\n    db.execute(q)\n"
    )}
    names, sinks, _ = run(files, "handler")
    assert names == {"escape", "run_query"}, names
    assert "db.execute" in sinks, sinks


@case("interprocedural across files: api.py -> db.py -> db.py")
def _():
    files = {
        "api.py": "import db\n\ndef handler(req):\n    return db.run_query(req)\n",
        "db.py": "def run_query(q):\n    return execute(q)\n\ndef execute(q):\n    return q\n",
    }
    idx, rf = build(files)
    ev = gather_evidence(idx["handler"][0], idx, rf)
    assert set(ev["helpers"]) == {"db.py::run_query", "db.py::execute"}, ev["helpers"]


@case("self.method across a class's methods, with the self offset handled")
def _():
    files = {"api.py": (
        "class R:\n"
        "    def clean(self, p):\n        return p.replace('..', '')\n\n"
        "    def read(self, name):\n        return self.load(self.clean(name))\n\n"
        "    def load(self, q):\n        return open(q)\n"
    )}
    idx, rf = build(files)
    ev = gather_evidence(idx["R.read"][0], idx, rf)
    keys = {k.split("::", 1)[1] for k in ev["helpers"]}
    assert keys == {"R.clean", "R.load"}, keys
    assert "open" in ev["sinks"], ev["sinks"]


@case("depth limit stops the walk")
def _():
    files = {"app.py": (
        "def a(x):\n    return b(x)\n\n"
        "def b(x):\n    return c(x)\n\n"
        "def c(x):\n    return d(x)\n\n"
        "def d(x):\n    return sink(x)\n\n"
        "def sink(x):\n    return x\n"
    )}
    names, _, _ = run(files, "a", max_depth=2)
    assert names == {"b", "c"}, names                  # d and sink are past the 2-hop limit


@case("helper cap stops the walk")
def _():
    defs = "".join(f"def h{i}(x):\n    return x\n\n" for i in range(1, 6))
    files = {"app.py": f"def f(x):\n    h1(x); h2(x); h3(x); h4(x); h5(x)\n\n{defs}"}
    _, _, ev = run(files, "f", max_helpers=3)
    assert len(ev["helpers"]) == 3, ev["helpers"]


@case("a cycle terminates and never re-adds the start function")
def _():
    files = {"app.py": "def a(x):\n    return b(x)\n\ndef b(x):\n    return a(x)\n"}
    names, _, _ = run(files, "a")
    assert names == {"b"}, names                        # a is the flagged fn, not its own helper


@case("no parameters -> no taint -> no helpers (triage keeps)")
def _():
    files = {"app.py": "def f():\n    return helper()\n\ndef helper():\n    return 1\n"}
    names, _, _ = run(files, "f")
    assert names == set(), names


# ---- source-seeded taint: a no-parameter handler whose input comes from a known
# SOURCE (input(), os.environ, sys.argv) still gets its chain walked; a plain module
# global does NOT (a global can hold anything, so tainting it would over-drop). ----

@case("no parameters but reads input() -> seeds from the source, walks the helper")
def _():
    files = {"app.py": (
        "def f():\n"
        "    name = input('path? ')\n"
        "    return open(clean(name))\n\n"
        "def clean(x):\n    return x.strip()\n"
    )}
    names, sinks, _ = run(files, "f")
    assert names == {"clean"}, names                    # input() is a source -> follow into clean
    assert "open" in sinks, sinks


@case("no parameters but reads os.environ -> seeds from the source, walks the helper")
def _():
    files = {"app.py": (
        "import os\n\n"
        "def f():\n"
        "    p = os.environ.get('TARGET', '')\n"
        "    return open(clean(p))\n\n"
        "def clean(x):\n    return os.path.basename(x)\n"
    )}
    names, sinks, _ = run(files, "f")
    assert names == {"clean"}, names
    assert "os.path.basename" in sinks, sinks           # the neutralising sink is reachable


@case("no parameters and a PLAIN global is NOT a source -> no helpers (no over-drop)")
def _():
    files = {"app.py": (
        "CURRENT = {}\n\n"
        "def f():\n"
        "    p = CURRENT.get('path', '')\n"
        "    return open(clean(p))\n\n"
        "def clean(x):\n    return x\n"
    )}
    names, _, _ = run(files, "f")
    assert names == set(), names                        # a global can hold anything -> not seeded


# ---- Flask request globals are the one (narrow) global we DO seed: a no-parameter view
# that reads request.form/args/json/... and passes it to a helper gets that chain walked;
# the bare `request` object itself is NOT seeded, only its caller-controlled members. ----

@case("no parameters but reads request.form.get -> seeds, walks the chain into the sink")
def _():
    files = {"app.py": (
        "import db\n\n"
        "def view():\n"
        "    name = request.form.get('name')\n"
        "    return run_query(build(name))\n\n"
        "def build(x):\n    return \"SELECT ... \" + x\n\n"
        "def run_query(q):\n    db.execute(q)\n"
    )}
    names, sinks, _ = run(files, "view")
    assert names == {"build", "run_query"}, names       # request.form is a source -> follow the chain
    assert "db.execute" in sinks, sinks

@case("request.get_json() through a subscript still carries taint into the helper")
def _():
    files = {"app.py": (
        "def view():\n"
        "    data = request.get_json()\n"
        "    return keygen(data['username'])\n\n"
        "def keygen(u):\n    return open('/tmp/' + u)\n"
    )}
    names, sinks, _ = run(files, "view")
    assert names == {"keygen"}, names                    # data['username'] carries data's taint
    assert "open" in sinks, sinks

@case("the bare `request` object is NOT a source (only its caller-controlled members)")
def _():
    files = {"app.py": (
        "def view():\n"
        "    r = request\n"
        "    return authenticate(r)\n\n"
        "def authenticate(x):\n    return x.headers\n"
    )}
    names, _, _ = run(files, "view")
    assert names == set(), names                         # bare `request` isn't seeded; only request.form/...


# ---- the safety checkpoint: the real sample's bundle is UNCHANGED from one-hop ----
def _real_sample():
    from chunker import walk_files, scan_file
    from resolver import assemble_repo_facts
    sample = ROOT.parent / "code-review-sample"   # the dedicated fixtures repo, a sibling of this project
    relabel = lambda p: str(Path(p).relative_to(sample))
    index, bindings = {}, {}
    files = walk_files(sample)
    for path in files:
        chunks, bindings[str(path)] = scan_file(path)
        for c in chunks:
            if c["kind"] == "code":
                index.setdefault(c["name"], []).append(
                    {"path": relabel(c["path"]), "name": c["name"], "code": c["code"]})
    return index, assemble_repo_facts(files, sample, bindings, relabel)


@case("real sample: read_export's bundle is exactly {safe_name} (unchanged)")
def _():
    idx, rf = _real_sample()
    ev = gather_evidence(idx["read_export"][0], idx, rf)
    assert set(ev["helpers"]) == {"api.py::safe_name"}, ev["helpers"]
    assert "open" in ev["sinks"], ev["sinks"]


@case("real sample: read_upload's bundle is exactly {normalize_path} (unchanged)")
def _():
    idx, rf = _real_sample()
    ev = gather_evidence(idx["read_upload"][0], idx, rf)
    assert set(ev["helpers"]) == {"api.py::normalize_path"}, ev["helpers"]


# ---- the nested cross-file pair: same shape, opposite verdict, 2 hops, across files ----
# fetch_report / fetch_avatar live in downloads.py; their helpers live in paths.py, so
# the walk only finds the real fix by crossing a file boundary AND going past hop one.

@case("real sample: fetch_report reaches the cross-file fix 2 hops deep (basename)")
def _():
    idx, rf = _real_sample()
    ev = gather_evidence(idx["fetch_report"][0], idx, rf)
    assert set(ev["helpers"]) == {"paths.py::build_safe_path",
                                  "paths.py::strip_traversal"}, ev["helpers"]
    assert "os.path.basename" in ev["sinks"], ev["sinks"]   # the neutralising sink triage cites


@case("real sample: fetch_avatar reaches the weak cross-file reshape (no basename -> kept)")
def _():
    idx, rf = _real_sample()
    ev = gather_evidence(idx["fetch_avatar"][0], idx, rf)
    assert set(ev["helpers"]) == {"paths.py::to_relative",
                                  "paths.py::collapse_slashes"}, ev["helpers"]
    assert "os.path.basename" not in ev["sinks"], ev["sinks"]  # nothing strips '../' -> stays


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
