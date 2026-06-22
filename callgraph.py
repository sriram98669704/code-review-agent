"""callgraph.py - taint-guided interprocedural expansion (the "call graph").

Triage judges whether a helper neutralises a finding. When the fix is ONE hop away
(the flagged function calls a sanitiser directly) reading that one helper is enough.
But the fix can sit DEEPER: api_handler -> build_query -> run_query -> db.execute.
A one-hop look inspects run_query, sees a clean-looking call, and misses that user
input reached the sink three hops up. This module follows the chain across files -
but only along the DATA, never blindly.

This IS the "call graph" people mean, built LAZILY: we never materialise a graph
object. Each edge is computed on demand with resolve() (which already maps a call to
the RIGHT file's def). And it is TAINT-GUIDED: starting from the flagged function's
parameters - the values that may carry user input - we follow only the calls that
actually RECEIVE a tainted value, exactly the slice CodeQL/Semgrep would walk. Three
guards keep it finite and demo-safe:

  * a visited set  - a function is expanded once, so a cycle (a <-> b) can't loop;
  * a depth limit  - stop following the chain past max_depth hops;
  * a helper cap   - stop once max_helpers sources have been gathered.

Everything here is pure AST + dict lookups: no API, no cost. The LLM still makes the
final call on whether a helper's sanitisation is SUFFICIENT (triage_finding); this
module only decides WHICH helper sources it gets to see.

Why taint-guidance is safe for the triage contract: the taint spread is CONSERVATIVE -
a fixpoint over assignments that ignores control flow, so it over-taints rather than
under-taints and never misses a real data path. Pruning a call that receives no
tainted value can only hide a helper that doesn't touch the risky value - which by
definition can't neutralise it - so it can never cause a wrong DROP, the one outcome
the triage contract forbids.
"""

import ast
from collections import deque

from resolver import resolve, _dotted, called_targets


def _qual(entry):
    """A function's unique label across the repo: 'path::name'. Two files can define
    the same name, so the name alone can't key the visited set or the helper bundle."""
    return f"{entry['path']}::{entry['name']}"


def _function(code):
    """The outermost function node in a chunk's source, or None. Chunks are dedented
    to the def, so a method parses as a top-level function here."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    return next((n for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)


def _params(fn):
    """(positional param names, has self/cls receiver, keyword-only/var param names).
    The positional order lets a tainted ARGUMENT position at a call site map to the
    parameter it binds inside the callee."""
    a = fn.args
    names = [p.arg for p in (a.posonlyargs + a.args)]
    receiver = bool(names) and names[0] in ("self", "cls")
    extra = [p.arg for p in a.kwonlyargs]
    if a.vararg:
        extra.append(a.vararg.arg)
    if a.kwarg:
        extra.append(a.kwarg.arg)
    return names, receiver, extra


# Known external-input SOURCES - the no-parameter equivalents of a tainted argument.
# A value read from one of these is treated as user-controlled, so a handler that takes
# NO parameters (stdin, an env var, argv, or an HTTP request) still gets its chain walked.
# Deliberately NARROW: an arbitrary module global is NOT a source - a global can hold
# anything, so tainting every global read would over-drop (the unsafe direction). The one
# global we DO seed is Flask's `request`, and only its caller-controlled members
# (form/args/json/...): it is a known framework input boundary, not an arbitrary value, so
# a no-parameter view that reads request.form and passes it to a helper gets that cross-
# function chain walked (this was the #1 item in README "What's next").
_TAINT_SOURCE_CALLS = {"input", "os.getenv", "os.environ.get", "request.get_json"}
_TAINT_SOURCE_PREFIXES = (
    "os.environ", "sys.argv",
    "request.form", "request.args", "request.values", "request.json",
    "request.data", "request.files", "request.cookies", "request.headers",
)


def _assignments(fn):
    """(targets, value) for every Assign/AnnAssign/AugAssign under fn - the edges taint
    spreads along."""
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            out.append((node.targets, node.value))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
            out.append(([node.target], node.value))
    return out


def _reads_source(value):
    """True if `value` reads from a known input source: input(), os.getenv(),
    os.environ[...]/os.environ.get(...), sys.argv, or a caller-controlled Flask request
    member (request.form/args/json/...). Used to seed taint where there is no parameter
    to seed from."""
    for sub in ast.walk(value):
        if isinstance(sub, ast.Call) and _dotted(sub.func) in _TAINT_SOURCE_CALLS:
            return True
        if isinstance(sub, (ast.Attribute, ast.Name)):
            d = _dotted(sub)
            if d and any(d == p or d.startswith(p + ".") for p in _TAINT_SOURCE_PREFIXES):
                return True
    return False


def seed_taint(code):
    """The taint seed for a flagged function: ALL its parameters (minus a self/cls
    receiver), PLUS any local assigned from a known input SOURCE (input(), os.environ,
    os.getenv, sys.argv, Flask request.form/args/...). The judge doesn't tell us WHICH
    value is user-controlled, so we
    treat every parameter - and every source read - as a potential source, conservative
    and the safe direction. Seeding from sources, not just parameters, lets a handler
    that takes NO parameters still get its helper chain walked."""
    fn = _function(code)
    if fn is None:
        return set()
    names, receiver, extra = _params(fn)
    pos = names[1:] if receiver else names
    seed = set(pos) | set(extra)
    for targets, value in _assignments(fn):        # a no-parameter handler seeds from here
        if _reads_source(value):
            for tgt in targets:
                seed |= _names_in(tgt)
    return seed


def _names_in(node):
    """Every bare Name referenced anywhere under `node`."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def tainted_calls(code, tainted):
    """Calls in `code` that RECEIVE tainted data, as (target, positions): `target` is
    a dotted call string (foo, db.get_user, self.clean); `positions` is the set of
    ARGUMENT indices carrying a tainted value, or None when a tainted value is passed
    by keyword/splat (position unknown -> seed all of the callee's params).

    Taint first spreads to any assignment whose value mentions a tainted name, to a
    fixpoint - order-independent and conservative (it may over-taint, never under-taint).
    """
    fn = _function(code)
    if fn is None:
        return []
    tainted = set(tainted)

    assigns = _assignments(fn)                         # (targets, value) we can taint through
    changed = True
    while changed:                                     # grow the tainted set to a fixpoint
        changed = False
        for targets, value in assigns:
            if _names_in(value) & tainted:
                for tgt in targets:
                    for nm in _names_in(tgt):
                        if nm not in tainted:
                            tainted.add(nm)
                            changed = True

    calls = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        target = _dotted(node.func)
        if not target:                                 # dynamic receiver (foo().bar()) -> unresolvable
            continue
        positions, by_keyword = set(), False
        for i, arg in enumerate(node.args):
            if isinstance(arg, ast.Starred):
                if _names_in(arg.value) & tainted:     # *args splat -> position unknown
                    by_keyword = True
            elif _names_in(arg) & tainted:
                positions.add(i)
        for kw in node.keywords:
            if _names_in(kw.value) & tainted:          # foo(x=tainted) / **kw -> position unknown
                by_keyword = True
        if positions or by_keyword:
            calls.append((target, None if by_keyword else positions))
    return calls


def _seed_callee(code, positions):
    """The taint seed for a callee, given the tainted ARGUMENT positions at the call
    site. `positions is None` (keyword/splat) -> seed every parameter. Maps an argument
    index to a parameter, accounting for a leading self/cls: an attribute call passes
    no receiver as an argument, so arg 0 binds the parameter AFTER self."""
    fn = _function(code)
    if fn is None:
        return set()
    names, receiver, extra = _params(fn)
    if positions is None:
        pos = names[1:] if receiver else names
        return set(pos) | set(extra)
    offset = 1 if receiver else 0
    seed = set()
    for i in positions:
        j = i + offset
        if 0 <= j < len(names):
            seed.add(names[j])
    return seed


def gather_evidence(entry, index, repo_facts, max_depth=3, max_helpers=12):
    """Follow the call chain from a flagged function `entry` along tainted data, and
    return the helper sources triage should reason over:

        {"helpers": {'path::name': source, ...},   # the bundle handed to the model
         "edges":   [('path::caller', 'path::callee'), ...],  # the resolved call path
         "sinks":   {dotted target, ...}}           # tainted calls that left the repo
                                                     # (db.execute, os.system, open) -
                                                     # the chain ends here, no source to read

    Only calls that receive a tainted value are followed (taint-guided); a visited set,
    `max_depth` hops, and `max_helpers` sources keep it finite. Pure AST + resolve():
    no API, no cost."""
    helpers, edges, sinks = {}, [], set()
    visited = {_qual(entry)}
    work = deque([(entry, seed_taint(entry["code"]), 0)])
    while work and len(helpers) < max_helpers:
        cur, tainted, depth = work.popleft()
        if depth >= max_depth:                         # past the depth limit -> stop following
            continue
        for target, positions in tainted_calls(cur["code"], tainted):
            callee = resolve(target, cur["path"], repo_facts, index, caller_name=cur["name"])
            if callee is None:                         # external/sink/undecidable - chain ends here
                sinks.add(target)
                continue
            cq = _qual(callee)
            edges.append((_qual(cur), cq))             # record the edge even if already gathered
            if cq in visited:                          # already have it (or a cycle back) -> don't re-walk
                continue
            visited.add(cq)
            helpers[cq] = callee["code"]
            if len(helpers) >= max_helpers:            # helper cap reached -> stop gathering
                break
            work.append((callee, _seed_callee(callee["code"], positions), depth + 1))
    return {"helpers": helpers, "edges": edges, "sinks": sinks}


# ---- the agent's OWN exploration, summarised for the dashboard ------------------
# gather_evidence above is the GUARANTEED, taint-guided pass triage runs. Separately,
# while reviewing, the agent may open a helper itself (read_function) - its own call,
# not guaranteed. This summary answers, for every helper the agent opened: which
# flagged function it belongs to, and whether the guaranteed pass would have reached
# it on its own. The interesting ones are the helpers the agent reached that the
# guaranteed pass does NOT - and WHY it doesn't, named against the real bounds.

def _reach_distance(entry, target_qual, index, repo_facts, max_hops=8):
    """Shortest number of resolved-call hops from `entry` to the function whose
    qualified 'path::name' is `target_qual`, following EVERY resolved call (NOT
    taint-gated), or None if it isn't reachable within `max_hops`.

    This is the un-gated companion to gather_evidence's taint-guided walk: a positive
    distance that exceeds the taint walk's depth bound means the helper sits ON the
    call chain but past the depth limit (so the guaranteed pass stopped short of it),
    which is exactly the distinction the exploration block needs to explain."""
    if _qual(entry) == target_qual:
        return 0
    visited = {_qual(entry)}
    work = deque([(entry, 0)])
    while work:
        cur, depth = work.popleft()
        if depth >= max_hops:                          # bound the search, like the taint walk
            continue
        for target in called_targets(cur["code"]):     # every call, not just tainted ones
            callee = resolve(target, cur["path"], repo_facts, index, caller_name=cur["name"])
            if callee is None:                          # unresolvable -> can't follow it
                continue
            cq = _qual(callee)
            if cq == target_qual:
                return depth + 1
            if cq in visited:                           # visited guard -> cycles can't loop
                continue
            visited.add(cq)
            work.append((callee, depth + 1))
    return None


def _attribute(qual, beyond, flagged_entries, taint_by_entry, index, repo_facts,
               max_depth, max_helpers):
    """For one helper the agent opened (`qual` = its 'path::name'), return
    (entry_qual, reason): the flagged function it belongs to, and - only when the
    GUARANTEED pass wouldn't reach it (`beyond` True) - WHY, named against the real
    bounds. reason is None for a helper the guaranteed pass also walks.

    The reason is read straight off program facts, never guessed:
      no-seed      - the entry takes no parameters and reads no tracked source, so the
                     taint walk has nothing to seed from and never starts (a plain
                     module global is deliberately not a source).
      depth        - the helper sits on the chain but more than `max_depth` hops down.
      width        - the entry's walk had already gathered `max_helpers` before it.
      off-path     - the helper is within reach but the tracked input never flows to it.
      unresolvable - reachable only past a call no static tool can pin down.
      not-on-chain - no flagged function calls it (the agent opened it on a hunch)."""
    if not beyond:                                       # the guaranteed pass walks it too:
        for q, helpers in taint_by_entry.items():        # name the entry whose walk reaches it
            if qual in helpers:
                return q, None
        return None, None
    best_q, best_d = None, None                          # nearest flagged caller (un-gated)
    for q, e in flagged_entries.items():
        d = _reach_distance(e, qual, index, repo_facts)
        if d is not None and d > 0 and (best_d is None or d < best_d):
            best_q, best_d = q, d
    if best_q is None:                                   # nothing flagged calls it
        return None, "not-on-chain"
    seed = seed_taint(flagged_entries[best_q]["code"])
    if not seed and not taint_by_entry.get(best_q):      # no parameters, no tracked source
        return best_q, "no-seed"
    if best_d > max_depth:                                # on the chain but past the depth bound
        return best_q, "depth"
    if len(taint_by_entry.get(best_q, ())) >= max_helpers:  # the walk filled its helper budget
        return best_q, "width"
    return best_q, "off-path"                             # reachable, but off the tracked value


def summarize_exploration(reads, flagged, index, repo_facts,
                          max_depth=3, max_helpers=12):
    """Summarise the helpers the agent opened ON ITS OWN (read_function), for the
    'agentic exploration' dashboard block. Pure AST + the live index - no API, no cost.

    `reads`   - the list of read_function results: {"name", "definitions":[{"path",
                "code"}, ...]}. Error dicts (a phantom name the agent asked for) carry
                no "definitions" and are skipped.
    `flagged` - the flagged functions whose chains triage walks (kept + dropped
                findings), each a dict with 'name' and 'path'. These are the entries.

    Returns:
      {"opened": [ {helper, helper_file, entry, entry_file, beyond_forced, reason},
                   ... ],                  # one per distinct helper the agent opened
       "forced_count":   helpers the guaranteed pass also reaches,
       "beyond_count":   helpers beyond the guaranteed pass (the ⭐ ones),
       "triage_reaches": how many distinct helpers the guaranteed pass reaches}

    Computed against the SAME bounds triage uses (gather_evidence's defaults), so the
    'beyond the guaranteed pass' split can never drift from what triage actually walks."""
    flagged_entries = {}                                 # qual -> index entry, de-duped
    for f in flagged:
        ents = index.get(f.get("name")) or []
        e = next((x for x in ents if x.get("path") == f.get("path")),
                 ents[0] if ents else None)
        if e is not None:
            flagged_entries[_qual(e)] = e

    taint_by_entry, triage_reaches = {}, set()           # what the guaranteed pass reaches
    for q, e in flagged_entries.items():
        helpers = set(gather_evidence(
            e, index, repo_facts, max_depth=max_depth, max_helpers=max_helpers)["helpers"])
        taint_by_entry[q] = helpers
        triage_reaches |= helpers

    opened, seen = [], set()
    for r in reads:
        defs = r.get("definitions")
        if not defs:                                     # error dict (phantom name) -> skip
            continue
        name = r.get("name")
        for d in defs:                                   # usually one; a name in two files -> each
            qual = f"{d.get('path')}::{name}"
            if qual in seen:                             # the agent opened the same helper twice
                continue
            seen.add(qual)
            beyond = qual not in triage_reaches
            entry_q, reason = _attribute(qual, beyond, flagged_entries, taint_by_entry,
                                         index, repo_facts, max_depth, max_helpers)
            entry_e = flagged_entries.get(entry_q) if entry_q else None
            opened.append({
                "helper": name,
                "helper_file": str(d.get("path", "")).rsplit("/", 1)[-1],
                "entry": entry_e["name"] if entry_e else None,
                "entry_file": str(entry_e["path"]).rsplit("/", 1)[-1] if entry_e else None,
                "beyond_forced": beyond,
                "reason": reason,
            })
    opened.sort(key=lambda o: (o["beyond_forced"], o["entry"] or "", o["helper"]))
    return {
        "opened": opened,
        "forced_count": sum(1 for o in opened if not o["beyond_forced"]),
        "beyond_count": sum(1 for o in opened if o["beyond_forced"]),
        "triage_reaches": len(triage_reaches),
    }
