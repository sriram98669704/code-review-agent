"""Offline tests for resolver.py - pure AST, no API, no cost. Run free, any time:

    /usr/bin/python3 tests/test_resolver.py

Each test is one row of the call-resolution table: does `resolve()` map a call to
the RIGHT file's function - and, just as important, does it return None (keep the
finding) instead of guessing when the call is genuinely undecidable.
"""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resolver import file_bindings, module_name, called_targets, resolve


def build(files):
    """Make an (index, repo_facts) pair from {path: source}, exactly the way the
    real index pass would: top-level defs and class methods go in the index; imports
    and defs go in the bindings; module map comes from the paths."""
    index, bindings, modules = {}, {}, {}
    for path, code in files.items():
        tree = ast.parse(code)
        bindings[path] = file_bindings(tree)
        modules[module_name(path)] = path
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                index.setdefault(node.name, []).append(
                    {"path": path, "name": node.name, "code": ast.get_source_segment(code, node)})
            elif isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        nm = f"{node.name}.{sub.name}"
                        index.setdefault(nm, []).append(
                            {"path": path, "name": nm, "code": ast.get_source_segment(code, sub)})
    return index, {"modules": modules, "files": bindings}


# Two files used across many cases. Note get_user is defined in BOTH - the classic
# name collision that bare-name matching gets wrong.
DB = (
    "def get_user(uid):\n    return q('select * from users where id=' + uid)\n\n"
    "def safe_name(p):\n    import os\n    return os.path.basename(p)\n"
)
API_OWN = (        # api.py that ALSO defines its own get_user
    "import db\n"
    "def get_user(req):\n    return req['user']\n\n"
    "def handler(req):\n    return db.get_user(req['id'])\n"
)


CASES = []
def case(name):
    def wrap(fn):
        CASES.append((name, fn))
        return fn
    return wrap


@case("bare call resolves to the caller's OWN file")
def _():
    idx, rf = build({"db.py": DB, "api.py": API_OWN})
    e = resolve("get_user", "api.py", rf, idx)
    assert e and e["path"] == "api.py", e


@case("qualified db.get_user() picks db's, NOT the caller's own get_user")
def _():
    idx, rf = build({"db.py": DB, "api.py": API_OWN})
    e = resolve("db.get_user", "api.py", rf, idx)
    assert e and e["path"] == "db.py", e          # the bug bare-matching can't avoid


@case("from-import: bare get_user() resolves to db when imported from db")
def _():
    api = "from db import get_user\n\ndef h(r):\n    return get_user(r['id'])\n"
    idx, rf = build({"db.py": DB, "api.py": api})
    e = resolve("get_user", "api.py", rf, idx)
    assert e and e["path"] == "db.py", e


@case("aliased module: d.get_user() where `import db as d`")
def _():
    api = "import db as d\n\ndef h(r):\n    return d.get_user(r['id'])\n"
    idx, rf = build({"db.py": DB, "api.py": api})
    e = resolve("d.get_user", "api.py", rf, idx)
    assert e and e["path"] == "db.py", e


@case("aliased name: gu() where `from db import get_user as gu`")
def _():
    api = "from db import get_user as gu\n\ndef h(r):\n    return gu(r['id'])\n"
    idx, rf = build({"db.py": DB, "api.py": api})
    e = resolve("gu", "api.py", rf, idx)
    assert e and e["path"] == "db.py", e


@case("bare call, unique in repo, resolves cross-file")
def _():
    api = "def h(p):\n    return safe_name(p)\n"   # safe_name lives only in db.py
    idx, rf = build({"db.py": DB, "api.py": api})
    e = resolve("safe_name", "api.py", rf, idx)
    assert e and e["path"] == "db.py", e


@case("self.method() resolves to the same class's method")
def _():
    api = ("class Reader:\n"
           "    def clean(self, p):\n        return p\n\n"
           "    def read(self, p):\n        return self.clean(p)\n")
    idx, rf = build({"api.py": api})
    e = resolve("self.clean", "api.py", rf, idx, caller_name="Reader.read")
    assert e and e["name"] == "Reader.clean", e


@case("LocalClass.method() resolves to the class in this file")
def _():
    api = ("class Reader:\n"
           "    def make(p):\n        return p\n\n"
           "def h(p):\n    return Reader.make(p)\n")
    idx, rf = build({"api.py": api})
    e = resolve("Reader.make", "api.py", rf, idx)
    assert e and e["name"] == "Reader.make", e


@case("ambiguous bare name (in two files, none local, no import) -> None")
def _():
    other = "def get_user(x):\n    return x\n"
    caller = "def h(x):\n    return get_user(x)\n"   # get_user in db.py AND other.py
    idx, rf = build({"db.py": DB, "other.py": other, "caller.py": caller})
    assert resolve("get_user", "caller.py", rf, idx) is None


@case("local var sharing a module's name (db = conn; db.run()) -> None, not the module")
def _():
    dbmod = "def run(q):\n    return q\n"                  # db.py HAS a run()
    api = "def h(conn):\n    db = conn\n    return db.run('x')\n"  # db is a LOCAL var, not imported
    idx, rf = build({"db.py": dbmod, "api.py": api})
    assert resolve("db.run", "api.py", rf, idx) is None    # must not guess db.py::run


@case("duck-typed x.save() on an unknown receiver -> None")
def _():
    api = "def h(x):\n    return x.save()\n"
    idx, rf = build({"db.py": DB, "api.py": api})
    assert resolve("x.save", "api.py", rf, idx) is None


@case("star import hides a bare name's origin -> None")
def _():
    api = "from db import *\n\ndef h(p):\n    return mystery(p)\n"
    idx, rf = build({"db.py": DB, "api.py": api})
    assert resolve("mystery", "api.py", rf, idx) is None


@case("called_targets keeps dotted forms and drops dynamic receivers")
def _():
    code = ("def h(d, k, x):\n"
            "    foo()\n"
            "    db.get_user(x)\n"
            "    self.clean(x)\n"
            "    make().bar()\n"        # call result receiver -> dropped
            "    handlers[k]()\n")      # subscript receiver   -> dropped
    t = called_targets(code)
    assert "foo" in t and "db.get_user" in t and "self.clean" in t, t
    assert not any("bar" in x or "handlers" in x for x in t), t


@case("called_targets handles keyword and *args/**kwargs calls")
def _():
    code = "def h(d):\n    foo(x, y=1)\n    bar(*args)\n    baz(**d)\n"
    t = called_targets(code)
    assert {"foo", "bar", "baz"} <= t, t


@case("multi-segment module: a.b.c.func() where `import a.b.c`")
def _():
    pkg = "def func(x):\n    return x\n"
    caller = "import a.b.c\n\ndef h(x):\n    return a.b.c.func(x)\n"
    idx, rf = build({"a/b/c.py": pkg, "caller.py": caller})
    e = resolve("a.b.c.func", "caller.py", rf, idx)
    assert e and e["path"] == "a/b/c.py", e


@case("aliased deep module: ps.func() where `import pkg.sub as ps`")
def _():
    pkg = "def func(x):\n    return x\n"
    caller = "import pkg.sub as ps\n\ndef h(x):\n    return ps.func(x)\n"
    idx, rf = build({"pkg/sub.py": pkg, "caller.py": caller})
    e = resolve("ps.func", "caller.py", rf, idx)
    assert e and e["path"] == "pkg/sub.py", e


@case("imported name used as a class: Thing.method() where `from m import Thing`")
def _():
    models = "class User:\n    def create(req):\n        return req\n"
    api = "from models import User\n\ndef h(r):\n    return User.create(r)\n"
    idx, rf = build({"models.py": models, "api.py": api})
    e = resolve("User.create", "api.py", rf, idx)
    assert e and e["name"] == "User.create" and e["path"] == "models.py", e


@case("self.method() picks the caller's OWN class when two classes share the name")
def _():
    api = ("class A:\n"
           "    def clean(self, p):\n        return p\n\n"
           "    def run(self, p):\n        return self.clean(p)\n\n"
           "class B:\n"
           "    def clean(self, p):\n        return p\n")
    idx, rf = build({"api.py": api})
    e = resolve("self.clean", "api.py", rf, idx, caller_name="A.run")
    assert e and e["name"] == "A.clean", e


@case("third-party module call (requests.get) -> None, not mis-resolved")
def _():
    caller = "import requests\n\ndef h(x):\n    return requests.get(x)\n"
    idx, rf = build({"caller.py": caller})
    assert resolve("requests.get", "caller.py", rf, idx) is None


@case("dotted stdlib call (os.path.basename) -> None")
def _():
    caller = "import os\n\ndef h(p):\n    return os.path.basename(p)\n"
    idx, rf = build({"caller.py": caller})
    assert resolve("os.path.basename", "caller.py", rf, idx) is None


@case("relative import is skipped: ambiguous bare name -> None (don't guess)")
def _():
    other = "def get_user(x):\n    return x\n"                 # get_user in db.py AND other.py
    caller = "from . import get_user\n\ndef h(x):\n    return get_user(x)\n"
    idx, rf = build({"db.py": DB, "other.py": other, "caller.py": caller})
    assert resolve("get_user", "caller.py", rf, idx) is None


@case("a name that exists only as a class method, called bare -> None")
def _():
    api = ("class A:\n"
           "    def clean(self, p):\n        return p\n\n"
           "def h(p):\n    return clean(p)\n")               # 'clean' has no top-level def
    idx, rf = build({"api.py": api})
    assert resolve("clean", "api.py", rf, idx) is None


@case("async helper resolves like any other def (unique in repo)")
def _():
    db = "async def fetch(x):\n    return x\n"
    api = "def h(x):\n    return fetch(x)\n"
    idx, rf = build({"db.py": db, "api.py": api})
    e = resolve("fetch", "api.py", rf, idx)
    assert e and e["path"] == "db.py", e


@case("attribute chain on a local var (a.b.c()) -> None")
def _():
    api = "def h(a):\n    return a.b.c(x)\n"
    idx, rf = build({"api.py": api})
    assert resolve("a.b.c", "api.py", rf, idx) is None


def main():
    passed = 0
    for name, fn in CASES:
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL  {name}\n      {e}")
        except Exception as e:                       # noqa: BLE001 - surface any error clearly
            print(f"ERROR {name}\n      {type(e).__name__}: {e}")
        else:
            print(f"pass  {name}")
            passed += 1
    print(f"\n{passed}/{len(CASES)} passed")
    sys.exit(0 if passed == len(CASES) else 1)


if __name__ == "__main__":
    main()
