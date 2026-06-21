"""resolver.py - map a call site to the ONE function it targets.

Triage needs the SOURCE of every helper a flagged function calls, so it can judge
whether a helper neutralises the risk. To pull the RIGHT source it must first
answer: when a function calls `db.get_user(...)`, which file's `get_user` is that?
Bare-name matching guesses; this resolver reads each file's imports and defs and
answers from facts.

Two halves, two moments:

  * build time (during indexing): file_bindings(tree) reads one parsed file's
    imports + top-level defs. The indexer collects these into `repo_facts` while it
    is ALREADY walking the file for chunks - no extra read, no extra parse.

  * triage time: resolve(target, caller_path, repo_facts, index) maps a call target
    to its def entry in the index, or returns None when the answer is genuinely
    undecidable (a duck-typed `x.save()`, a dynamic getattr, a star import). None
    means "don't guess" - the finding is kept, which is the safe direction.

The contract is the whole point: resolve MORE than bare-name matching, but never
GUESS. When unsure, return None.
"""

import ast
from pathlib import Path


# ---- build time: facts pulled from the parse the chunker already did ------------

def module_name(rel_path):
    """Dotted module name for a repo-relative path: 'pkg/util.py' -> 'pkg.util',
    'pkg/__init__.py' -> 'pkg', 'db.py' -> 'db'. Lets a qualified call like
    `pkg.util.f()` find the file that defines `f`. `rel_path` is a pathlib.Path."""
    parts = list(Path(rel_path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def file_bindings(tree):
    """From ONE parsed module, return what its names point to - per-file SYNTAX only,
    no cross-file resolution yet (resolve() connects these across files later):

        {
          "imports": { local_name: ("module", dotted)        # import db [as d]
                                  | ("name", dotted, orig) }, # from db import f [as g]
          "defs":    { top-level def/class names },
          "star":    bool,   # `from x import *` seen -> a bare name may come from x
        }
    """
    imports, defs, star = {}, set(), False
    for node in tree.body:
        if isinstance(node, ast.Import):                  # import db / import db as d / import a.b
            for alias in node.names:
                if alias.asname:                          # import a.b.c as d  -> d points at a.b.c
                    imports[alias.asname] = ("module", alias.name)
                else:                                     # import a.b.c       -> binds top name `a`
                    imports[alias.name.split(".")[0]] = ("module", alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:                                # relative import (from . import x): we
                continue                                  # don't resolve package-relative yet -> skip
            module = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    star = True
                    continue
                local = alias.asname or alias.name
                imports[local] = ("name", module, alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.add(node.name)
        elif isinstance(node, ast.ClassDef):
            defs.add(node.name)
    return {"imports": imports, "defs": defs, "star": star}


def assemble_repo_facts(files, root, bindings_by_path, relabel=None):
    """Stitch per-file bindings into the repo-wide fact book the resolver reads:

        { "modules": { dotted_module_name: path },   # which file IS this module
          "files":   { path: <file_bindings result> } }

    Pure path math + a relabel - no parsing here (the bindings were already pulled
    from the chunker's parse). `relabel(path_str) -> str` rewrites on-disk paths to
    the clean display paths the index uses, so a caller_path matches; identity if
    None. `files` is the list of pathlib.Paths; `root` is the repo root Path."""
    relabel = relabel or (lambda p: p)
    modules = {}
    for path in files:
        rel = Path(path).relative_to(root)
        modules[module_name(rel)] = relabel(str(path))
    return {
        "modules": modules,
        "files": {relabel(p): b for p, b in bindings_by_path.items()},
    }


# ---- triage time: turn a call into the function it targets ----------------------

def called_targets(code):
    """Every call target in `code`, as a dotted string: `foo`, `db.get_user`,
    `self.clean`. Returns a set. A target whose receiver is not a plain name chain
    (e.g. `foo().bar()`, `handlers[k]()`) is dropped - it is dynamic and no static
    tool can resolve it."""
    targets = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return targets
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            dotted = _dotted(node.func)
            if dotted:
                targets.add(dotted)
    return targets


def _dotted(node):
    """Reconstruct a dotted name from a Name/Attribute chain, or None if the base is
    not a plain name (a call result, subscript or literal -> dynamic receiver)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _entry(index, name, path=None):
    """The ONE index entry for `name`, disambiguated by `path` when given. With no
    path: the entry if the name is unique in the repo, else None (don't guess)."""
    entries = index.get(name)
    if not entries:
        return None
    if path is not None:
        return next((e for e in entries if e.get("path") == path), None)
    return entries[0] if len(entries) == 1 else None


def resolve(target, caller_path, repo_facts, index, caller_name=None):
    """Map a call `target` (from called_targets) made inside `caller_path` to the
    index entry of the function it calls, or None when it is undecidable.

    `caller_name` is the name of the function the call sits in (e.g. 'Foo.bar'); it
    lets `self.method` resolve to a method of the SAME class. Returns the def entry
    (a dict with 'path' and 'code') or None - None keeps the finding, never guesses.
    """
    if not target:
        return None
    segs = target.split(".")
    files = repo_facts.get("files", {})
    modules = repo_facts.get("modules", {})
    caller = files.get(caller_path) or {"imports": {}, "defs": set(), "star": False}
    imports = caller["imports"]

    # ---- bare call: foo() ----
    if len(segs) == 1:
        name = segs[0]
        if name in caller["defs"]:                    # defined in this file -> this file's wins
            return _entry(index, name, caller_path)
        b = imports.get(name)
        if b and b[0] == "name":                      # from db import foo  ->  db's foo
            path = modules.get(b[1])
            e = _entry(index, b[2], path) if path else None
            if e:
                return e
        if caller["star"]:                            # could have come from `import *` -> can't tell
            return None
        return _entry(index, name)                    # unique-in-repo, else None

    # ---- qualified call: head.rest ----
    head, rest = segs[0], segs[1:]
    func = rest[-1]                                   # the final segment is the called name
    b = imports.get(head)

    # self.method() inside a method -> a method of the same class, same file
    if head == "self" and len(rest) == 1 and caller_name and "." in caller_name:
        cls = caller_name.rsplit(".", 1)[0]
        return _entry(index, f"{cls}.{func}", caller_path)

    # head is an imported MODULE (import db / import db as d) -> module's function
    if b and b[0] == "module" and len(rest) == 1:
        path = modules.get(b[1])
        e = _entry(index, func, path) if path else None
        if e:
            return e

    # longest dotted prefix that names a repo module: a.b.c.f -> module a.b.c, def f.
    # Gated on head being imported AS A MODULE here, so a local variable that happens
    # to share a module's name (db = connect(); db.execute()) is never mistaken for
    # the module db.py - that would be a guess, and a guess can wrongly drop a finding.
    if b and b[0] == "module":
        for i in range(len(segs) - 1, 0, -1):
            path = modules.get(".".join(segs[:i]))
            if path:
                tail = ".".join(segs[i:])             # may be Class.method within that module
                e = _entry(index, tail, path) or _entry(index, segs[-1], path)
                if e:
                    return e

    # head is an imported NAME used as a class: from db import Thing; Thing.method()
    if b and b[0] == "name" and len(rest) == 1:
        path = modules.get(b[1])
        e = _entry(index, f"{b[2]}.{func}", path) if path else None
        if e:
            return e

    # head is a class defined in THIS file: Foo.method()
    if head in caller["defs"] and len(rest) == 1:
        e = _entry(index, f"{head}.{func}", caller_path)
        if e:
            return e

    # head is a local variable / parameter / unknown receiver -> duck-typed -> keep
    return None
