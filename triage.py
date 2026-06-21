"""Triage - the disconfirmation pass that kills false positives.

The judge raises a finding by looking at ONE function in isolation, so it cannot
see what the helpers that function calls actually do - it flags a path built from
user input whether or not a helper already neutralises the attack. Triage does
what a careful researcher does AFTER discovery: it tries to KILL each finding.

Two things make it trustworthy where a single all-in-one pass flinches:

  1. Fresh context. The model that raised a finding is reluctant to delete its own
     work; a separate context, whose ONLY job is to disprove the finding, has no
     such attachment - so it actually pulls the trigger on a false positive.

  2. Evidence discipline. We don't ask the model to recall what a helper does - we
     mechanically read the flagged function's AST, pull the FULL SOURCE of every
     helper it calls from the index, and hand that real code over. To DROP a
     finding the model must quote the exact line that neutralises the attack; if it
     can't, the finding stays. When unsure, it keeps - a false drop hides a real
     bug, which is the one outcome we never want.
"""

import json

from llm import chat
from security import DETERMINISTIC_RULES
from callgraph import gather_evidence

TRIAGE_SYSTEM = (
    "You are a security triage reviewer. A first-pass judge raised a finding by "
    "looking at ONE function in isolation, so it could not see what the helper "
    "functions that function calls actually do. Your job is the OPPOSITE of the "
    "judge's: try to KILL the finding.\n"
    "You are given the finding, the flagged function, and the FULL SOURCE of the "
    "helpers it calls. Decide whether any helper genuinely removes the SPECIFIC "
    "risk the finding describes.\n"
    "Rules:\n"
    "- To DROP, you MUST name the helper and quote the exact line in it that "
    "neutralises the attack (e.g. a call that strips '../' from a path). With no "
    "real neutralising line you may NOT drop.\n"
    "- A helper that only reshapes the input without removing the risk (e.g. swaps "
    "backslashes for slashes but leaves '../' intact) does NOT neutralise it - KEEP.\n"
    "- If in doubt, KEEP. A wrong drop hides a real vulnerability.\n"
    "Answer in JSON."
)


def triage_finding(finding, func_code, helpers, api_key=None):
    """Ask a FRESH model context to kill ONE finding, given the flagged function and
    the source of the helpers it calls. Returns the verdict dict:
    {"decision": "keep"|"drop", "helper": str, "line": str, "reason": str}."""
    helper_block = "\n\n".join(f"# {name}\n{src}" for name, src in helpers.items())
    user = (
        f"FINDING: {finding.get('rule_id')} "
        f"(severity {finding.get('severity')}) at line {finding.get('line')}\n"
        f"What the judge said: {finding.get('explanation')}\n\n"
        f"FLAGGED FUNCTION ({finding.get('name')}):\n{func_code}\n\n"
        f"HELPERS IT CALLS:\n{helper_block}\n\n"
        'Return JSON: {"decision": "keep" or "drop", '
        '"helper": "<helper name, or empty>", '
        '"line": "<the exact neutralising line, or empty>", '
        '"reason": "<one sentence>"}'
    )
    resp = chat(
        messages=[{"role": "system", "content": TRIAGE_SYSTEM},
                  {"role": "user", "content": user}],
        api_key=api_key,
        response_format={"type": "json_object"},
        temperature=0,
        seed=0,                               # same dice every run -> repeatable
    )
    data = json.loads(resp.choices[0].message.content)
    return {
        "decision": "drop" if data.get("decision") == "drop" else "keep",
        "helper": data.get("helper", ""),
        "line": data.get("line", ""),
        "reason": data.get("reason", ""),
    }


def triage(findings, index, repo_facts, api_key=None, on_step=None):
    """Re-check every finding against the helpers its function REACHES, and split them
    into (kept, dropped). `index` is the agent's function index (name -> entries with
    'code' and 'path'); `repo_facts` is the resolver's fact book (imports + module map)
    built in the same index pass. Together they let triage gather the right helper
    source straight from memory - no file touch, no re-embed.

    For each finding we don't just read the one function's direct calls. gather_evidence
    follows the call chain across files along the TAINTED data (seeded from the flagged
    function's parameters), so a fix that sits several hops deep is still seen, while a
    branch the input never reaches is left out. Resolution is import-aware - db.get_user()
    finds db.py's def, not a same-named one elsewhere - and refuses to guess when a call
    is undecidable, so a wrong file's source can never disconfirm a real finding.

    A finding is KEPT untouched (never sent to the model) when there's nothing to
    disconfirm: it's a deterministic rule (already certain), its function isn't in the
    index (top-level code), or no helper sits on a tainted path out of it. Everything
    else gets the disconfirmation pass; a dropped finding carries the triage verdict
    (the cited neutralising line) under 'triage' so the UI can show why it was overruled."""
    kept, dropped = [], []
    for f in findings:
        if f.get("rule_id") in DETERMINISTIC_RULES:    # AST-certain -> nothing to disprove
            kept.append(f)
            continue
        entries = index.get(f.get("name"))
        if not entries:                                # top-level / unknown -> keep
            kept.append(f)
            continue
        entry = next((e for e in entries if e.get("path") == f.get("path")), entries[0])
        evidence = gather_evidence(entry, index, repo_facts)
        helpers = evidence["helpers"]                  # {path::name: source} along tainted data
        if not helpers:                                # nothing we can disconfirm against -> keep
            kept.append(f)
            continue
        if on_step:                                    # show the path: helpers, then any external sink
            via = ", ".join(sorted(helpers))
            if evidence["sinks"]:
                via += " -> " + ", ".join(sorted(evidence["sinks"]))
            on_step(f"triaging {f.get('path')}::{f.get('name')} "
                    f"({f.get('rule_id')}) via {via}")
        verdict = triage_finding(f, entry["code"], helpers, api_key=api_key)
        if verdict["decision"] == "drop":
            dropped.append({**f, "triage": verdict})
        else:
            kept.append(f)
    return kept, dropped
