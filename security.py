"""The policy check (the judge) - security, GRC, and guardrail rules.

One chunk at a time, STATELESS: every chunk gets a fresh LLM call. Instead of
showing it ALL the rules, we RETRIEVE the most relevant ones first - the rules
are embedded into a small Chroma vector store, and for each chunk we pull the
nearest few rules from each category. The rules come from every file in
policies/, each tagged with its category (security / grc / guardrail). The model
can only report issues matching a rule we SHOWED it; a grounding guard then
THROWS AWAY any finding citing a rule id outside that retrieved set, so the model
can never invent an issue out of thin air.
"""

import ast
import json
from pathlib import Path

import chromadb

from chunker import walk_files, chunk_file
from llm import chat, embed, embed_many

RULES_PER_CATEGORY = 5        # how many rules from each category to show a chunk

# Rules answered by mechanically reading the AST, not by asking the model.
# no-bare-except is a pure syntax question ("does this except body surface the
# error or not?") - a parser answers it with certainty, so judging it requires
# zero interpretation. Asking the LLM bought us nothing but run-to-run wobble
# (it disagreed with itself on two IDENTICAL except blocks). These rules are
# still loaded, embedded, and retrieved exactly like every other rule - they
# just never reach the LLM judge; scan_chunk() routes them to a Python check
# instead and the result comes back in the same finding shape.
DETERMINISTIC_RULES = {"no-bare-except"}

# Calls that count as "surfacing" an error inside an except body.
_SURFACE_CALL_NAMES = {"print", "log", "debug", "info", "warning", "warn",
                        "error", "exception", "critical"}

POLICY_DIR = Path(__file__).parent / "policies"
CATEGORY = {"owasp": "security", "grc": "grc", "guardrails": "guardrail"}


def _load_rules():
    """Load every policies/*.json and tag each rule with its category."""
    rules = []
    for path in sorted(POLICY_DIR.glob("*.json")):
        category = CATEGORY.get(path.stem, path.stem)
        for rule in json.loads(path.read_text()):
            rule["category"] = category
            rules.append(rule)
    return rules


RULES = _load_rules()
RULE_BY_ID = {r["id"]: r for r in RULES}
CATEGORIES = sorted({r["category"] for r in RULES})


def _build_policy_store(api_key=None):
    """Put every rule in a Chroma collection so we can find the rules whose
    MEANING is closest to a chunk of code.

    We embed each rule's title + description and tag it with its category, so we
    can retrieve PER category - a big category (security) gets pruned to its most
    relevant few, while a small one (grc, guardrail) is never starved."""
    store = chromadb.EphemeralClient()
    coll = store.get_or_create_collection(
        name="policy_store", metadata={"hnsw:space": "cosine"})
    texts = [f"{r['title']}. {r['description']}" for r in RULES]
    coll.add(
        ids=[r["id"] for r in RULES],
        embeddings=embed_many(texts, api_key=api_key),
        metadatas=[{"category": r["category"]} for r in RULES],
    )
    return coll


_policy_store = None      # the embedded rule store, built lazily on first review


def _get_policy_store(api_key=None):
    """Build the policy store on first use, then reuse it. Building it lazily (not
    at import) means importing this module makes NO API call: a deployed UI with
    no .env can load fine and wait for the user to paste a key, which then pays
    for the one-time rule embedding. The rules are fixed, so we embed them once."""
    global _policy_store
    if _policy_store is None:
        _policy_store = _build_policy_store(api_key)
    return _policy_store


def retrieve_rules(chunk, k=RULES_PER_CATEGORY, api_key=None, vec=None):
    """Find the rules most relevant to ONE chunk: the k nearest rules from EACH
    category. Returns a list of rule dicts - only these get shown to the model.

    `vec` is the chunk's embedding if it was already computed. The index pass
    embeds each chunk ONCE and shares that vector with the duplicate check too, so
    it passes the vector in here to skip a redundant embed; omit it and we embed
    the chunk ourselves."""
    store = _get_policy_store(api_key)
    if vec is None:
        vec = embed(chunk["code"], api_key=api_key)
    rules = []
    for cat in CATEGORIES:
        n = min(k, sum(1 for r in RULES if r["category"] == cat))
        res = store.query(
            query_embeddings=[vec], n_results=n, where={"category": cat})
        rules.extend(RULE_BY_ID[rid] for rid in res["ids"][0])
    return rules


JUDGE_SYSTEM = (
    "You are a code reviewer checking rules across categories such as security "
    "vulnerabilities, compliance (GRC), and engineering guardrails. You are "
    "given the RULES grouped by category and ONE chunk of code with line "
    "numbers. Report only issues in the code that match one of the rules. For "
    "each issue cite the rule's exact id and the line number shown, and ALWAYS "
    "give a 'fix': a minimal corrected version of just the flawed line(s) - not "
    "the whole function. Even when the real remedy needs more than one line (e.g. "
    "adding an authorization check), show the key corrected or added line(s) so a "
    "reader sees the direction - never leave 'fix' empty. Do "
    "not invent rules. Read each rule's SAFE note and do NOT flag code that the "
    "note calls safe. If the code is clean, report nothing. Respond in JSON."
)


def _number_lines(chunk):
    """Prefix each code line with its REAL file line number, so the model can
    cite a line that matches the actual file (chunk['start'] is the offset)."""
    lines = chunk["code"].splitlines()
    return "\n".join(f"{chunk['start'] + i}: {line}" for i, line in enumerate(lines))


def _rules_block(rules):
    """Render the given rules, grouped under a header per category."""
    by_cat = {}
    for r in rules:
        by_cat.setdefault(r["category"], []).append(r)
    blocks = []
    for cat in sorted(by_cat):
        lines = "\n".join(
            f"- {r['id']} ({r['severity']}): {r['description']}" for r in by_cat[cat]
        )
        blocks.append(f"[{cat.upper()} RULES]\n{lines}")
    return "\n\n".join(blocks)


def _surfaces_error(handler):
    """True if an except HANDLER's body surfaces the error somewhere - a raise,
    or a call that looks like logging/printing it. Walks the WHOLE body (not
    just the top line) so a surfacing call nested in an if/with still counts.
    A commented-out print is invisible to the AST, so it correctly counts as
    NOT surfacing - the error still vanishes at runtime."""
    for node in ast.walk(ast.Module(body=handler.body, type_ignores=[])):
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name in _SURFACE_CALL_NAMES:
                return True
    return False


def _check_bare_except(chunk):
    """The deterministic version of no-bare-except: walk the chunk's own AST,
    find every broad except (bare `except:` or `except Exception`/`BaseException`),
    and flag the ones whose body doesn't surface the error. Specific exception
    types (`except ValueError`, `except InvalidKey`, ...) are always safe."""
    rule = RULE_BY_ID["no-bare-except"]
    findings = []
    tree = ast.parse(chunk["code"])
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            is_broad = handler.type is None or (
                isinstance(handler.type, ast.Name) and handler.type.id in {"Exception", "BaseException"})
            if is_broad and not _surfaces_error(handler):
                findings.append({
                    "path": chunk["path"],
                    "name": chunk["name"],
                    "line": chunk["start"] + handler.lineno - 1,
                    "rule_id": rule["id"],
                    "rule_title": rule["title"],
                    "category": rule["category"],
                    "severity": rule["severity"],
                    "explanation": "Broad except whose body never surfaces the "
                                    "error (no raise, log, or print) - it silently vanishes.",
                    "fix": "except Exception as exc:\n"
                           "    logging.exception(exc)  # surface it\n"
                           "    raise",
                })
    return findings


def judge_chunk(chunk, rules, api_key=None):
    """Judge ONE chunk against rules ALREADY retrieved for it. This is the half of
    the old scan_chunk that forms an opinion: rules that need interpretation go to
    the LLM (grounded against only these retrieved rules); rules that are pure
    syntax (see DETERMINISTIC_RULES) are answered straight from the AST, no model
    call.

    Splitting retrieval (above) from judgement (here) is what lets the index pass
    retrieve + cache a chunk's rules ONCE, then let the agent trigger the judgement
    later - via judge() - without re-embedding or re-retrieving anything."""
    llm_rules = [r for r in rules if r["id"] not in DETERMINISTIC_RULES]

    findings = []
    if "no-bare-except" in {r["id"] for r in rules}:
        findings.extend(_check_bare_except(chunk))

    if not llm_rules:                          # nothing left for the model to judge
        return findings

    user = (
        f"RULES:\n{_rules_block(llm_rules)}\n\n"
        f"CODE ({chunk['path']} - {chunk['name']}):\n{_number_lines(chunk)}\n\n"
        'Return JSON of this shape: '
        '{"findings": [{"line": <int>, "rule_id": "<id>", '
        '"severity": "<high|medium|low>", "explanation": "<one sentence>", '
        '"fix": "<corrected or added code line(s) - never empty>"}]}'
    )
    resp = chat(
        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user", "content": user}],
        api_key=api_key,
        response_format={"type": "json_object"},
        temperature=0,
        seed=0,                               # same dice every run -> repeatable
    )
    raw = json.loads(resp.choices[0].message.content).get("findings", [])

    valid = {r["id"]: r for r in llm_rules}   # only retrieved rules are grounded
    for f in raw:
        rule_id = f.get("rule_id")
        if rule_id not in valid:              # GROUNDING GUARD: drop invented rules
            continue
        rule = valid[rule_id]
        findings.append({
            "path": chunk["path"],
            "name": chunk["name"],
            "line": f.get("line"),
            "rule_id": rule_id,
            "rule_title": rule["title"],
            "category": rule["category"],
            "severity": f.get("severity"),
            "explanation": f.get("explanation"),
            "fix": f.get("fix", ""),
        })
    return findings


def scan_chunk(chunk, api_key=None):
    """Retrieve a chunk's rules and judge it in one shot - the original one-call
    path, kept for the CLI below and any direct caller. The agent uses the two
    halves separately: retrieve_rules during indexing, judge_chunk on demand."""
    rules = retrieve_rules(chunk, api_key=api_key)
    return judge_chunk(chunk, rules, api_key=api_key)


if __name__ == "__main__":
    sample = Path(__file__).parent.parent / "code-review-sample"
    all_findings = []
    for path in walk_files(sample):
        for chunk in chunk_file(path):
            all_findings.extend(scan_chunk(chunk))

    print(f"{len(all_findings)} findings:\n")
    for f in sorted(all_findings, key=lambda x: (x["category"], str(x["path"]), x["line"] or 0)):
        rel = Path(f["path"]).relative_to(sample)
        print(f"  [{f['category']:9}] {rel}:{f['line']}  ({f['severity']}) {f['rule_id']}")
        print(f"       {f['explanation']}")
