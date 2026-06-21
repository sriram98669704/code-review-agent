"""The code-review agent.

The LLM is the brain (the manager). We give it a workspace and a mission, then let
it drive the review: it indexes the repo once, lists the functions, judges each one,
reads a helper's source when a verdict depends on code it can't see, overrules that
verdict if the helper changes the picture, and finally writes the report. That
decide -> act -> observe cycle is what makes this an agent.

Two layers, on purpose:

  Layer 1 - index_repo(): the cheap STRUCTURAL pass, done ONCE. Walk every file and
      split it into chunks. For a FUNCTION, just record where it lives (name -> file
      + source) so the agent has a checklist and can read or review it on demand. For
      TOP-LEVEL code (imports, constants like API_KEY = "..."), settle it right here -
      it has no helper to investigate and the agent never reviews it by name, so
      there is nothing for the agent to add. It does NOT embed, judge, or dedup the
      functions: that all happens later, per function, in ONE place.

  Layer 2 - the tools the agent drives over that workspace: list_functions() to see
      what to review, and judge(name) - the per-function step that does EVERYTHING
      for one function at once: embed its code a single time, retrieve the rules
      closest to it, check it against the functions seen so far for duplicates, and
      return the rule verdict. read_function(name) pulls a helper's source when a
      verdict depends on it.

Keeping the per-function verdict out of Layer 1 is the whole point: it's what leaves
the agent a real job to do, instead of a finished report to reword.
"""

import json
from datetime import datetime
from pathlib import Path

from chunker import walk_files, scan_file
from resolver import assemble_repo_facts
from security import retrieve_rules, judge_chunk
from duplicates import check_duplicate, new_pile, drop_pile
from triage import triage
from callgraph import summarize_exploration
from llm import chat, embed, AGENT_MODEL

HONESTY_LINE = (
    "This is an LLM-assisted review (Python files only); complements tools "
    "like bandit/semgrep, doesn't replace them. Finding counts may vary "
    "slightly between runs."
)

# Generous safety caps so an enormous repo (or a model that never stops calling
# tools) can't run the loop - or the bill - away. Both are far above anything a
# normal review needs (the test sample is ~30 functions), so they never trip on
# our own runs; they only bound the pathological case. When a cap is hit the run
# still finishes and the report SAYS what was skipped - it never truncates silently.
MAX_FUNCTIONS = 200                       # most functions handed to the agent per run
MAX_ROUNDS = 4 * MAX_FUNCTIONS + 40       # hard ceiling on agent loop turns (the seatbelt)


# The "function index" - a phone book the index pass fills as it walks: name -> list
# of entries, one per definition of that name. Each entry records where a function
# lives and its source, so the agent's tools can list it, read it, or review it
# WITHOUT touching a file again; judge() also caches its own review on the entry, so
# a function is embedded, judged, and added to the duplicate pile exactly once. The
# value is a LIST because two files can define the same name - one file can't, since
# it can't define a name twice - so we keep EVERY match and let the agent pick the
# right one by reading them. Built fresh at the start of each index_repo call and
# cleared when an agent run ends, so a shared instance never leaks one visitor's
# workspace into the next.
_INDEX = {}

# The resolver's fact book, built in the SAME walk that fills _INDEX (no second pass):
# { "modules": {dotted name -> file}, "files": {file -> imports/defs} }. Triage reads
# it to map a call like `db.get_user()` to the RIGHT file's def. Reset with the index.
_REPO_FACTS = {}

# The duplicate pile: a Chroma vector store that judge() streams functions into as it
# reviews them. Duplicate detection is RELATIONAL - each function is compared against
# the ones reviewed before it - so the pile has to outlive a single judge() call and
# live here, beside the index. Created by index_repo, freed when the run ends.
_PILE = None


def _reset_workspace():
    """Clear the index and free the duplicate pile - a clean slate for a new run, and
    no residue left behind when one ends."""
    global _PILE
    _INDEX.clear()
    _REPO_FACTS.clear()
    if _PILE is not None:
        drop_pile(_PILE)
        _PILE = None


def _explore(reads, review):
    """Summarise the helpers the agent opened ON ITS OWN (read_function) for the
    dashboard's agentic-exploration block. MUST run before the workspace is reset, so
    it reads the still-live index/facts here, not at render time. The entries are the
    findings triage walks chains for - the survivors plus the dropped false positives.

    This is a VIEW, not the verdict: if summarising ever fails it returns None and the
    dashboard simply omits the block (it already guards on exploration is not None) -
    a presentation detail must never sink an otherwise-complete review."""
    try:
        return summarize_exploration(reads, review["findings"] + review["dropped"],
                                     _INDEX, _REPO_FACTS)
    except Exception:                              # noqa: BLE001 - degrade the view, keep the review
        return None


# ---- Layer 1: the index pass - map the functions, settle the top-level code -----

def index_repo(directory, api_key=None, on_step=None, label=None):
    """Walk the repo ONE FILE AT A TIME and build the workspace the agent reviews.
    For a FUNCTION, just record it in the index (name -> file + source) so the agent
    has a checklist and can read or review it later - no embedding or judging here.
    For TOP-LEVEL code (imports, constants like API_KEY = "..."), settle it right now:
    it has no helper to investigate and the agent never reviews it by name, so there
    is nothing for the agent to add.

    Returns the workspace summary - counts and the already-settled top-level findings.
    The per-function work (embed, rules, duplicates, verdict) is withheld for judge().
    Nothing is written to disk.

    `on_step(msg)` is an optional progress callback (default None = silent).
    `label` is a clean display name (e.g. 'owner/repo') used in progress messages
    and in each finding's path, so the UI never shows the throwaway clone path."""
    global _PILE
    def step(msg):
        if on_step:
            on_step(msg)

    root = Path(__file__).parent / directory
    shown = label or directory
    step(f"walking repo: {shown}")
    _reset_workspace()                          # fresh index + pile for this run
    _PILE = new_pile()                          # judge() streams functions into this
    module_findings = []

    def relabel(path_str):
        # Cached paths and findings carry the absolute on-disk path (a throwaway
        # temp dir for a cloned repo). Rewrite it to a clean, repo-relative path
        # prefixed with the label, so the UI shows "owner/repo/db.py" not
        # "/var/folders/.../db.py".
        try:
            rel = Path(path_str).relative_to(root)
        except ValueError:
            return path_str
        return f"{shown}/{rel}" if label else str(rel)

    files = walk_files(root)
    bindings_by_path = {}                        # file -> imports/defs, for the resolver
    for path in files:                          # one FILE at a time
        step(f"scanning file: {path.relative_to(root)}")
        chunks, bindings_by_path[str(path)] = scan_file(path)  # one read, one parse: chunks + bindings
        for chunk in chunks:                    # one CHUNK (AST node) at a time
            if chunk["kind"] == "code":         # a function -> just map it; judge reviews it later
                step(f"mapping {chunk['name']}")
                _INDEX.setdefault(chunk["name"], []).append({
                    "path": chunk["path"], "name": chunk["name"],
                    "code": chunk["code"], "start": chunk["start"],
                })
            else:                               # top-level code: no helper to read, so
                step(f"checking top-level code in {path.relative_to(root)}")
                vec = embed(chunk["code"], api_key=api_key)             # embed ONCE...
                rules = retrieve_rules(chunk, api_key=api_key, vec=vec)  # ...reuse it
                module_findings.extend(         # settle it now (iron coverage)
                    judge_chunk(chunk, rules, api_key=api_key))
    step("done")

    for entries in _INDEX.values():             # clean every cached path...
        for e in entries:
            e["path"] = relabel(e["path"])
    _REPO_FACTS.update(                          # ...build the resolver's fact book on the SAME
        assemble_repo_facts(files, root, bindings_by_path, relabel))  # relabelled paths
    for f in module_findings:                   # ...and the top-level findings we produced.
        if "path" in f:
            f["path"] = relabel(f["path"])
        if str(f.get("name", "")).startswith("<"):   # raw AST tag (<Assign>, <Import>): give it a
            f["name"] = "top-level code"             # clean label for the cards AND the verdict

    return {
        "scanned": shown,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "files": len(files),
        "functions": sum(len(v) for v in _INDEX.values()),
        "module_findings": module_findings,
        "note": HONESTY_LINE,
    }


# ---- Layer 2: the tools the agent drives over the workspace ---------------------

def list_functions(api_key=None, on_step=None):
    """List every function name in the indexed repo - the agent's review checklist.

    `api_key` is accepted but unused (no API call) so the agent loop can call every
    tool the same way - see run_agent."""
    if on_step:
        on_step("listing functions")
    if not _INDEX:                              # SEATBELT: asked before any index ran
        return {"error": "No workspace yet - run index_repo first."}
    names = sorted(_INDEX)                      # sorted -> the cap is reproducible
    if len(names) > MAX_FUNCTIONS:              # huge repo: hand over a generous prefix...
        return {"functions": names[:MAX_FUNCTIONS],
                "note": (f"This repository has {len(names)} functions; reviewing the "
                         f"first {MAX_FUNCTIONS} (alphabetical). State in your report "
                         f"that {len(names) - MAX_FUNCTIONS} function(s) were not "
                         f"reviewed because the per-run cap was reached.")}
    return {"functions": names}


def judge(name, api_key=None, on_step=None):
    """Review ONE function, by NAME - the per-function step where everything about it
    happens in one place. For each definition of the name: embed its code ONCE, use
    that one vector to retrieve the rules closest to it AND to check it against the
    functions reviewed so far for duplicates, then judge it against those rules.
    Returns the rule findings and any duplicate match.

    The verdict looks at that function ALONE. If its correctness depends on a helper
    it calls, the agent is expected to read that helper (read_function) and decide
    for itself - this verdict is a starting point, not the final word. If a name lives
    in more than one file we review EVERY definition. Reviewing the same name twice
    returns the first review (cached on the entry), so a function is embedded, judged,
    and added to the duplicate pile exactly once."""
    global _PILE
    if not _INDEX:                              # SEATBELT: asked before any index ran
        return {"error": "No workspace yet - run index_repo first."}
    if _PILE is None:                           # the index exists but the pile is gone - e.g. an
        _PILE = new_pile()                      # overlapping run's cleanup freed it. Rebuild so
                                                # the duplicate check can't crash on None.count().
    entries = _INDEX.get(name)
    if not entries:                             # phantom name: don't emit a "reviewing"
        return {"error": f"function '{name}' not found in the indexed repo."}
    findings, duplicates = [], []
    for e in entries:
        if "review" not in e:                   # review each definition exactly once
            if on_step:                         # one step per real definition (file::name),
                on_step(f"reviewing {e['path']}::{e['name']}")  # so the count matches the tree
            chunk = {"path": e["path"], "name": e["name"],
                     "code": e["code"], "start": e["start"]}
            vec = embed(chunk["code"], api_key=api_key)             # embed ONCE...
            rules = retrieve_rules(chunk, api_key=api_key, vec=vec)  # ...for the rules...
            dup = check_duplicate(chunk, _PILE, api_key=api_key, vec=vec)  # ...and the dups
            e["review"] = {
                "findings": judge_chunk(chunk, rules, api_key=api_key),
                "duplicate": dup,
            }
        findings.extend(e["review"]["findings"])
        if e["review"]["duplicate"]:
            duplicates.append(e["review"]["duplicate"])
    return {"name": name, "findings": findings, "duplicates": duplicates}


def read_function(name, api_key=None, on_step=None):
    """Read ONE function's full source by NAME, from the index built during indexing.

    The investigation tool: the agent pulls in a helper's code when a finding's
    correctness depends on what that helper actually does (e.g. "does the helper this
    code calls actually validate the input?"). A plain dict lookup - no file is
    re-parsed - that resolves ACROSS files, so the agent supplies only a name, not a
    path. If a name lives in more than one file we return EVERY match and let the
    agent pick the right one by reading them.

    `api_key` is accepted but unused (no API call) so the agent loop can call every
    tool the same way - see run_agent."""
    if not _INDEX:                              # SEATBELT: asked before any index ran
        return {"error": "No workspace yet - run index_repo first."}
    defs = _INDEX.get(name)
    if not defs:                                # phantom name: don't emit a "reading" step
        return {"error": f"function '{name}' not found in the indexed repo."}
    if on_step:                                 # carry the file so the UI can show file::name
        on_step(f"reading function: {defs[0]['path']}::{name}")
    return {"name": name,
            "definitions": [{"path": d["path"], "code": d["code"]} for d in defs]}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "index_repo",
            "description": "Index a code repository ONCE into a workspace you can then "
                           "query. Maps every function to its file and settles risks in "
                           "top-level code (imports, module constants); returns those "
                           "findings and counts. It does NOT judge the functions or find "
                           "duplicates - use list_functions and judge for that.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Folder to review."}
                },
                "required": ["directory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_functions",
            "description": "List every function index_repo found - your checklist of "
                           "what to judge. Call index_repo first.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "judge",
            "description": "Review ONE function by name: check it against the rules "
                           "(security, compliance/GRC, guardrails) AND flag it if it "
                           "duplicates a function reviewed earlier. The risk verdict "
                           "looks at that function ALONE - if its risk depends on a "
                           "helper it calls, read that helper and decide for yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Function to judge."}
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_function",
            "description": "Read the full source of ONE function by its NAME, from the "
                           "functions index_repo found. Use this to settle a verdict "
                           "that depends on what another function it calls actually "
                           "does - not for every function. If the same name exists in "
                           "more than one file you get every match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the function to read - "
                                                                "often a helper the judged "
                                                                "function calls."},
                },
                "required": ["name"],
            },
        },
    },
]

TOOL_IMPLS = {
    "index_repo": index_repo,
    "list_functions": list_functions,
    "judge": judge,
    "read_function": read_function,
}

SYSTEM = (
    "You are a code-review agent. You review a repository and report its risks.\n\n"
    "Your tools build a workspace and then query it:\n"
    "- index_repo(directory): index the repo once; maps every function to its file "
    "and settles risks in top-level code, already done.\n"
    "- list_functions(): list every function, so you know what to review.\n"
    "- judge(name): review ONE function - its rule-based risk verdict, plus whether "
    "it duplicates a function reviewed earlier.\n"
    "- read_function(name): the full source of ONE function.\n\n"
    "Index the repo, then judge every function so nothing is skipped. But a judge "
    "verdict sees ONLY that one function, so it cannot tell a real risk from one a "
    "helper already neutralises - it will flag both the same way. So whenever a "
    "finding depends on what another function does (a path built by a helper, input "
    "passed through a validator, a value returned by a checker), that verdict is "
    "PROVISIONAL: read that helper with read_function, then decide for yourself - "
    "DROP the finding if the helper genuinely makes the code safe, KEEP it if the "
    "helper does not actually remove the risk. The final call is yours; say which "
    "findings you overruled and why.\n\n"
    "Then write the risk report as clean Markdown so it renders with clear "
    "structure - never one dense block of text. Follow this format exactly:\n"
    "- Do NOT open with a title or heading such as 'Final Risk Report'.\n"
    "- One '#### ' heading per category that has findings: '#### Security', "
    "'#### Compliance (GRC)', '#### Guardrails'.\n"
    "- Within a category, put a bold severity label on its OWN line - "
    "'**High severity**', '**Medium severity**', '**Low severity**' - then one "
    "Markdown bullet ('- ') per finding beneath it.\n"
    "- Each bullet on ONE line, shaped: '- **<function>** (<file>) - <one-line "
    "issue>. *Fix:* <short fix>.'\n"
    "- If any finding was overruled by triage, add a final '#### Overruled (false "
    "positives)' section with one bullet each, naming the function, the rule, and the "
    "helper line that makes it safe.\n"
    "- If duplicate code was reported, add a '#### Duplicates' section with one bullet "
    "per pair.\n"
    "- End with a '---' line, then a single bold line: '**Overall:** <one-sentence "
    "verdict>'.\n"
    "Keep every bullet to one line - this is a readable summary; the detailed cards "
    "are shown above it."
)


def run_agent(mission, api_key=None, on_step=None):
    """Run the decide -> act -> observe loop and RETURN its result (no printing), so
    a UI can render it. Returns:

        {"report": the agent's plain-text risk verdict,
         "review": {scanned, timestamp, findings, dropped, duplicates, note} for the
                   cards. The agent collects raw judge() verdicts, then a TRIAGE pass
                   re-checks each against the helpers its function calls and splits
                   them: findings = the survivors, dropped = false positives killed
                   with the cited neutralising line (under each one's 'triage' key),
         "reads":  list of read_function results - helpers the agent pulled in,
         "exploration": summary of those reads for the dashboard - which helpers the
                   agent opened on its OWN, and which ones it reached that the
                   guaranteed triage walk does not (see callgraph.summarize_exploration)}

    Decisions stream live through on_step ("agent thinking", "agent calling ...") so
    a caller can show the loop as it happens. The CLI (see __main__) prints the
    returned report itself."""
    def step(msg):
        if on_step:
            on_step(msg)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": mission},
    ]
    review = {"scanned": None, "timestamp": None,
              "findings": [], "dropped": [], "duplicates": [], "note": HONESTY_LINE}
    reads = []           # functions the agent pulled in via read_function
    judged = set()       # functions already folded into review["findings"]

    try:
        for _ in range(MAX_ROUNDS):                # <-- THE AGENT LOOP (bounded: see the else)
            step("agent thinking")
            resp = chat(messages=messages, tools=TOOLS, tool_choice="auto",
                        api_key=api_key, model=AGENT_MODEL)
            msg = resp.choices[0].message
            messages.append(msg)                   # remember what the agent said

            if not msg.tool_calls:                 # no tool wanted -> discovery is done
                # TRIAGE: a fresh pass tries to KILL each raised finding, reading the
                # real source of the helpers its function calls. review["findings"] is
                # the raw judge verdicts; here we split them into survivors + dropped
                # false positives. This is where the override actually fires - the
                # discovery agent, attached to findings it raised, tends to flinch.
                kept, dropped = triage(review["findings"], _INDEX, _REPO_FACTS,
                                       api_key=api_key, on_step=on_step)
                review["findings"], review["dropped"] = kept, dropped
                report = msg.content
                if dropped:                        # make the report agree with triage
                    killed = "\n".join(
                        f"- {d['name']} ({d.get('rule_id')}): {d['triage']['reason']}"
                        for d in dropped)
                    messages.append({
                        "role": "user",
                        "content": ("A triage pass re-checked your findings against the "
                                    "helpers each function calls and DROPPED these as "
                                    "false positives:\n" + killed + "\n\nWrite your final "
                                    "report in the SAME Markdown format described earlier, "
                                    "EXCLUDING these findings from their categories, and add "
                                    "the '#### Overruled (false positives)' section "
                                    "explaining each drop."),
                    })
                    step("agent thinking")
                    report = chat(messages=messages, api_key=api_key,
                                  model=AGENT_MODEL).choices[0].message.content
                return {"report": report, "review": review, "reads": reads,
                        "exploration": _explore(reads, review)}

            for call in msg.tool_calls:            # run each tool it asked for
                name = call.function.name
                args = json.loads(call.function.arguments)
                step(f"agent calling {name}({args})")
                result = TOOL_IMPLS[name](**args, api_key=api_key, on_step=on_step)
                if name == "index_repo":           # counts + settled top-level findings
                    review["scanned"] = result.get("scanned")
                    review["timestamp"] = result.get("timestamp")
                    review["findings"].extend(result.get("module_findings", []))
                elif name == "judge":              # fold each function's verdict + dups in once
                    nm = result.get("name")
                    if nm and nm not in judged:
                        judged.add(nm)
                        review["findings"].extend(result.get("findings", []))
                        review["duplicates"].extend(result.get("duplicates", []))
                elif name == "read_function":      # remember each helper it pulled in
                    reads.append(result)
                messages.append({                  # hand the result back to the agent
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                })
        else:
            # SEATBELT: the loop ran MAX_ROUNDS turns and the agent STILL wanted tools
            # (a return above is the only normal exit). Stop calling tools, triage what
            # we have, and force an honest final report instead of looping - or billing -
            # forever. Reached only in the pathological case, never on a normal review.
            step(f"step limit ({MAX_ROUNDS}) reached - forcing final report")
            kept, dropped = triage(review["findings"], _INDEX, _REPO_FACTS,
                                   api_key=api_key, on_step=on_step)
            review["findings"], review["dropped"] = kept, dropped
            drop_note = ""
            if dropped:
                drop_note = ("\n\nA triage pass also DROPPED these as false positives - "
                             "exclude them and add a '#### Overruled (false positives)' "
                             "section:\n" + "\n".join(
                                 f"- {d['name']} ({d.get('rule_id')}): {d['triage']['reason']}"
                                 for d in dropped))
            messages.append({
                "role": "user",
                "content": (f"You have reached the {MAX_ROUNDS}-step limit for this review. "
                            "Stop investigating and write your final report NOW, in the same "
                            "Markdown format described earlier, from what you judged so far. "
                            "Add a line stating the review hit its step limit and may be "
                            "incomplete." + drop_note),
            })
            report = chat(messages=messages, api_key=api_key,
                          model=AGENT_MODEL).choices[0].message.content
            return {"report": report, "review": review, "reads": reads,
                    "exploration": _explore(reads, review)}
    finally:
        _reset_workspace()                         # leave no workspace residue behind (index + pile)


if __name__ == "__main__":
    sample = Path(__file__).parent.parent / "code-review-sample"   # the fixtures repo (a sibling)
    out = run_agent(
        f"Review the repository at '{sample}' and report its risks.",
        on_step=lambda m: print(f"  · {m}"),       # show the loop live in the terminal
    )
    print("\n=== AGENT REPORT ===\n")
    print(out["report"])

    dropped = out["review"]["dropped"]
    print(f"\n=== TRIAGE DROPPED {len(dropped)} FALSE POSITIVE(S) ===")
    for d in dropped:                              # the override, shown plainly
        t = d["triage"]
        print(f"  - {d['name']} ({d['rule_id']}): overruled via {t['helper']} "
              f"-> {t['line']!r}\n    {t['reason']}")

    expl = out.get("exploration")                  # the helpers the agent opened ON ITS OWN
    if expl and expl.get("opened"):                # (read_function) - same data the dashboard
        print(f"\n=== AGENT OPENED {len(expl['opened'])} HELPER(S) ON ITS OWN "  # block draws
              f"({expl['beyond_count']} beyond the guaranteed triage walk) ===")
        for o in expl["opened"]:
            star = " *BEYOND*" if o["beyond_forced"] else ""
            why = f" [{o['reason']}]" if o["reason"] else ""
            entry = f"{o['entry']}()" if o["entry"] else "(no flagged caller)"
            print(f"  - {entry} -> {o['helper']}() in {o['helper_file']}{star}{why}")

    print(f"\n{HONESTY_LINE}")
