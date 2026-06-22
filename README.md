# Code Review Agent

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![Built with Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b) ![Powered by OpenAI](https://img.shields.io/badge/powered%20by-OpenAI-412991) ![Live review](https://img.shields.io/badge/live%20review-ephemeral-lightgrey)

**Can an LLM review code the way a human reviewer would — read each function, flag the real risks, and explain why?**

**[Live dashboard](https://ai-llm-code-review-agent.streamlit.app/)** — paste a GitHub repo URL and your own OpenAI key, see it run. · **[Demo video](https://drive.google.com/file/d/14TeN77E1aj-7FHNyH_5ylyWUTgnJmD6c/view?usp=drive_link)** — walkthrough of a full run.

Static scanners match patterns; they don't reason about what a function actually does. This is an LLM *agent* that reads Python one function at a time and flags security holes, compliance (GRC) issues, weak guardrails, and duplicated logic — then explains every finding in plain English. It's a real agent, not a single prompt: the model **indexes** the repo, **judges** each function against the rules, **reads a helper's source** when a verdict depends on it, and a separate **triage** pass **overrules** any finding a helper proves harmless — so you get real risks, not false alarms. It complements tools like bandit/semgrep — it doesn't replace them.

---

## At a glance

```bash
venv/bin/streamlit run dashboard.py     # live review in the browser
venv/bin/python agent.py                # the same review from the CLI
```

- **Works on any public Python repo** — paste a GitHub URL into the dashboard (or pass a path to `run_agent()`) and only its `.py` files are fetched, scanned, and deleted; the bundled [`code-review-sample`](https://github.com/sriram98669704/code-review-sample) is just the default demo input, not something the tool is built around
- **Two layers, on purpose** — a deterministic *index* pass walks the repo once (one file in memory at a time), maps every function to its file and settles top-level code, and **withholds the per-function verdict** — embedding, rule-retrieval, duplicate detection, and judgement all happen later, per function, in one place — which is exactly what leaves the agent a real job to do
- **A real agent loop** — the LLM indexes the repo, judges every function, and pulls in a helper's source when a verdict depends on it (decide → act → observe)
- **Triage overrules false positives** — a separate, skeptical pass re-reads the helper a flagged function calls and drops the finding *only* if it can cite the exact line that removes the risk, so a real bug is never silently dropped
- **Four lenses** — Security, Compliance (GRC), Guardrails, and duplicate detection, every finding explained in plain English and paired with a suggested fix
- **Retrieval over brute force** — two ChromaDB vector stores (cosine similarity): rules are retrieved top-k per chunk (RAG, not "show it everything"), duplicates are found via nearest-neighbour search instead of comparing every function pair
- **Bring-your-own-key** — env-first locally, a per-session BYOK panel when deployed; keys never touch disk, logs, or `os.environ`
- **Ephemeral by design** — a live run is shown on screen only and is never written to disk

---

## How it works

### The agent loop

The LLM is the brain. We hand it a workspace and a mission, then let it drive: it indexes the repo, judges each function, reads a helper's source when a verdict depends on it, and writes the verdict — and a triage pass overrules the false positives. That **decide → act → observe** cycle is what makes this an agent rather than a script.

**Why a hand-written loop, not a framework.** The loop is written directly against the model's tool-calling API rather than wrapped in an agent framework (LangChain, an SDK `Agent`/`run()`). That's deliberate, not missing: a framework's `run()` gives the *same* architecture — it would not make this "more of an agent" — but driving the loop directly gives **finer control over each step than a managed `run()` does**, which is exactly what made the two-pass design (a paranoid first review, then a separate triage pass that overrules it) and the taint-guided evidence-gathering straightforward to express, with every step explicit and visible in the timeline. That control is the trade: a framework would cut boilerplate, but it would not change what the system does or make it more correct. The genuine production gaps are orthogonal to framework choice — see [What's next](#whats-next).

```
mission ─▶ index_repo() — walk every file once, map each function to its file,
              and settle top-level code → returns a WORKSPACE.
              On purpose, it WITHHOLDS the per-function verdict.
              ↓
          list_functions() — the checklist of what to review
              ↓
          judge(name) for each function — embed it once, retrieve its rules,
              check it for duplicates, return its rule verdict (in ISOLATION)
              ↓
          does a verdict hinge on what a helper it calls actually does?
              ├─ yes ─▶ read_function(helper) ─▶ read its source
              └─ no
              ↓
          TRIAGE — a fresh, skeptical pass re-checks each finding against the
          helpers its function calls, and DROPS the ones a helper proves safe
              ↓
          write the risk report (consistent with triage)
```

**Two layers, on purpose.** Layer 1 (`index_repo`) is the cheap structural pass done *once* — walk every file, chunk it, and record where each function lives (name → file + source). Top-level code (imports, constants like `API_KEY = "..."`) has no helper to investigate and the agent never reviews it by name, so it's settled right here. Layer 1 returns a workspace (the function index + those top-level findings) but **withholds the verdict on the functions** — and does no embedding, rule-retrieval, or duplicate detection on them. That withholding is the whole point: it leaves the agent a real job instead of a finished report to reword. Layer 2 is the tools the agent drives over that workspace — `list_functions()`, `judge(name)` (the per-function step that does *everything* for one function at once: embed its code a single time, retrieve the rules closest to it, check it against the functions reviewed so far for duplicates, and return its rule verdict), and `read_function(name)` to pull a helper's source by name (an instant index lookup, resolved across files; if a name exists in more than one file, the agent gets every match).

Only **one file is held in memory at a time** during indexing. The duplicate "pile" — each reviewed function's embedding vector — persists across the whole run, because finding duplicates inherently means comparing a function against all the ones reviewed before it; `judge` streams every function it reviews into it. The pile is freed when the run ends.

### Triage — overruling false positives

A `judge` verdict sees **one function in isolation**, so it can't tell a real risk from one a helper already neutralises — it flags both the same way. Picture two functions that each build a file path from user input: the judge flags **both** for path traversal, and the two findings come back *identical*. The only way to tell them apart is to read the helper each one calls and reason about whether it actually sanitises — one helper might run `os.path.basename` (strips `../` → safe), while the other only swaps slashes (leaves `../` → still vulnerable). Same finding on the surface; opposite verdict once you read one level down.

Triage is the pass that does this, and it's deliberately **separate** from discovery, for two reasons a single all-in-one pass can't satisfy:

- **Fresh context.** The model that raised a finding is reluctant to delete its own work. A separate context whose *only* job is to disprove the finding has no such attachment — so it actually pulls the trigger on a false positive.
- **Evidence discipline.** We don't ask the model to recall what a helper does — we mechanically read the flagged function's AST, follow the calls it makes, and hand over the **full source** of the real helpers behind them. Two static passes make that evidence trustworthy. **Resolution** ([`resolver.py`](resolver.py)) maps each call to the **right file** by reading the caller's imports — `db.get_user()`, a `from db import get_user`, an aliased `import db as d`, or a `self.method()` each resolve to the exact definition they name; a call it genuinely can't pin down (a duck-typed `x.save()`, or a bare name that collides across files — the same helper name defined in two different modules) is left **unresolved rather than guessed**, so triage never disconfirms from the wrong source. **Reach** ([`callgraph.py`](callgraph.py)) follows the chain *past the first hop* — `f` → `build` → `run` → sink — but only along **tainted data**: starting from the flagged function's parameters it expands only the calls that actually receive a risky value (the slice CodeQL/Semgrep would walk), bounded by a visited set, a depth limit, and a helper cap. So a fix three hops deep is still seen, and a branch the input never reaches is never pulled in. To **drop** a finding the model must quote the exact line that neutralises the attack; if it can't, the finding **stays**. When in doubt it keeps — a wrong drop hides a real bug, the one outcome we never want.

The result is the agent doing what a plain scanner can't: the safe one is **dropped** (citing the exact helper line that neutralises it), while the genuinely vulnerable one is **kept** — separated by reading the helper, not by a hardcoded rule. ([`triage.py`](triage.py))

### Retrieval, not brute force — two vector stores (ChromaDB)

Both checks lean on the same trick: embed things into vectors, store them in an **in-memory ChromaDB collection** (`hnsw:space=cosine`), and query by **cosine similarity** instead of comparing against everything every time. Why that matters: showing the LLM your *entire* rule set on every chunk, or comparing every chunk against every prior chunk, doesn't scale as either pile grows — retrieval keeps each call cheap and fast regardless of how big the policy set or repo gets.

- **Security / GRC / guardrail check — retrieval-augmented, not brute-force.** Every rule in `policies/*.json` is embedded once, up front. For each code chunk, we embed the chunk and pull only the **top-k nearest rules per category** (`RULES_PER_CATEGORY = 5` in [`security.py`](security.py)) — not the whole rulebook. That keeps the prompt small and on-topic even if the policy set grows to hundreds of rules. A grounding guard then discards any finding that cites a rule ID outside that retrieved set, so the model can never invent a violation.
- **Duplicate check — nearest-neighbour, not pairwise.** As each function is reviewed it's embedded and queried against the growing "pile" of every earlier reviewed function's vector, asking only for the **single closest match** (`n_results=1` in [`duplicates.py`](duplicates.py)). Only that top match — if its cosine similarity clears `THRESHOLD = 0.80` — gets sent to the LLM to confirm it's a real duplicate (not just a coincidental match). Confirmed duplicates are tagged **exact** (byte-for-byte identical body) or **similar** (functionally the same even when reworded — e.g. renamed variables or reordered lines; matched by meaning via embeddings, not a fixed list of edits), so a reviewer knows whether it's a literal copy or a near-twin. This avoids an O(n²) LLM-comparison blow-up as the repo grows; ChromaDB's index does the nearest-neighbour search instead.

In both cases the LLM only ever sees a small, relevant slice — retrieved by vector search — never the entire rule set or the entire history of functions.

---

## The dashboard

```bash
venv/bin/streamlit run dashboard.py
```

A single live-review page — no tabs, no stored history:

1. **Set your key** — auto-detected from `.env` locally, or pasted into the BYOK panel on a deployed instance.
2. **Paste a public GitHub repo link** — e.g. `https://github.com/owner/repo`. Add a GitHub `/tree/<branch>/<subdir>` suffix to scope the review to one folder — e.g. `https://github.com/fportantier/vulpy/tree/master/bad` reviews only `bad/`, not the sibling `good/`. Only its `.py` files are fetched — via the GitHub API, one file at a time, into a temp directory ([`fetcher.py`](fetcher.py)) — then scanned and deleted; local and deployed runs behave identically. See [Fetching a repo](#fetching-a-repo).
3. **Run** — the agent runs the full **decide → act → observe** loop: it indexes the repo, judges every function, pulls in a helper when a verdict depends on it, runs a triage pass that overrules the false positives, and writes a plain-English risk verdict. Progress streams live as a decision timeline (index → judge → investigate → triage → write) with the index tree (every file and function) nested inside it. The page then shows a severity breakdown (counts + bar chart) up top, then the surviving findings, an **Overruled** section listing each false positive triage dropped (with the neutralising line it cited), the duplicates, and finally the agent's narrative verdict as a plain-English wrap-up — each finding paired with a suggested fix (the judge returns it in the same call, so it costs no extra API request).

A run is **ephemeral**: its result lives in memory, renders on screen, and is never saved to disk — so a shared public instance never leaks one visitor's scan to the next.

### Fetching a repo

Only the repo's `.py` files are pulled — **never** its `.git` history or any non-Python file. `fetched_repo()` ([`fetcher.py`](fetcher.py)) lists the file tree in one API call, keeps the Python blobs, and downloads each into a throwaway temp dir that's deleted after the scan. A `GITHUB_TOKEN` env var raises the rate limit from 60 to 5000 requests/hour; without one, small public repos still work, and if the API path can't run (rate-limited with no token, a transient error) it falls back to a shallow `git clone`. The dashboard names which path ran on every review — *Fetched N .py file(s) via the GitHub API*, or a note that the API was unavailable and it cloned instead — so the fetch path is never invisible.

A true **never-touches-disk, RAM-only** fetch was considered and rejected: the index, the duplicate pile, and the resolver all hold every function in memory for the whole run anyway, so streaming files saves nothing downstream — while it would force a rewrite of the chunker, indexer, and resolver. The temp-dir fetch already skips the `.git`/non-Python bulk (the actual saving) at a fraction of the risk.

---

## Bring Your Own Key

A live review makes real, paid OpenAI calls, so it needs a key. The security model is strict and deliberate:

- **Env-first.** If a local `.env` supplies `OPENAI_API_KEY`, the dashboard uses it directly and the BYOK panel stays hidden. The panel only appears when no key is in the environment — e.g. on the deployed app, which has no `.env`.
- **Pasted keys live in browser-session memory only** — never written to disk, logs, or environment variables, and never shared between visitors.
- **Never written to `os.environ`** (which is process-global and shared across sessions — writing a key there could leak it between visitors). The key flows as an explicit function argument straight to the OpenAI SDK call.
- **Anything key-shaped is redacted** (`«redacted-key»`) before it can ever reach the UI or a log — defence-in-depth against provider errors that echo a partial key.
- Keys vanish when the tab closes, and a **Clear key** button wipes the pasted key on demand.

Key resolution is pure and side-effect-free, in [`byok.py`](byok.py).

**Deployment hardening.** [`.streamlit/config.toml`](.streamlit/config.toml) disables telemetry (`gatherUsageStats`), keeps XSRF/CORS protection on (Streamlit's secure defaults), and turns off `showErrorDetails` so an uncaught crash shows a plain "something went wrong" instead of a traceback with internal paths on the public page. No server-side secret ever needs configuring — the deployed app has no `.env`, so it runs purely on BYOK, with no `st.secrets` to manage or leak.

---

## Quickstart

### Prerequisites

- Python 3.9+
- An OpenAI API key

### Install

```bash
git clone <your-repo-url>
cd code-review-agent

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# fill in OPENAI_API_KEY
```

`.env` is gitignored and never committed. For a deployed app, no `.env` exists — the dashboard falls back to the BYOK panel, scoped to that one visitor's browser session.

### Run the dashboard

```bash
venv/bin/streamlit run dashboard.py
```

### Run the CLI agent

```bash
venv/bin/python agent.py
```

Runs the full decide → act → observe loop and prints a plain-text risk report to the terminal. With no arguments it reviews the bundled [`code-review-sample`](https://github.com/sriram98669704/code-review-sample) fixtures repo (a sibling of this project) as a demo; to review your own code, pass a path to `run_agent()` (see [Using it in code](#using-it-in-code)) or paste a GitHub URL into the dashboard.

---

## Using it in code

Run the full agent — it drives the tools itself, overrules false positives, and writes a narrative report:

```python
from agent import run_agent

# Returns {"report": verdict text,
#          "review": {findings, dropped, duplicates, ...},  # cards: survivors + overruled
#          "reads":  helpers the agent pulled in,
#          "exploration": which of those helpers the agent opened on its OWN,
#                         and which it reached beyond the guaranteed triage walk}.
out = run_agent("Review the repository at '/path/to/repo' and report its risks.",
                api_key="sk-...")

review = out["review"]
print(f"{len(review['findings'])} findings, "
      f"{len(review['dropped'])} overruled, "
      f"{len(review['duplicates'])} duplicates")

for f in review["findings"]:                       # the risks that survived triage
    print(f"[{f['severity']}] {f['path']}:{f['name']} — {f['rule_title']}")

for d in review["dropped"]:                        # false positives triage overruled
    t = d["triage"]
    print(f"DROPPED {d['name']} ({d['rule_id']}) via {t['helper']}: {t['line']!r}")

print(out["report"])
```

The two layers are also callable directly if you want the substrate without the agent loop: `index_repo(directory)` builds the workspace (the function map + top-level findings), then `judge(name)` (rule verdict + any duplicate) / `read_function(name)` query it. See [`agent.py`](agent.py).

---

## Project structure

```
code-review-agent/
├── agent.py          # Agent loop (decide → act → observe): index_repo + list_functions + judge + read_function tools
├── triage.py         # the disconfirmation pass — kills false positives, citing the neutralising line
├── resolver.py       # import-aware call resolution — maps db.get_user() to the RIGHT file's def (never guesses)
├── callgraph.py      # taint-guided interprocedural expansion — follows the call chain to deep helpers
├── chunker.py        # walk_files + scan_file — AST chunking (+ per-file import facts), one file at a time
├── security.py       # retrieve_rules + judge_chunk — RAG-grounded security / GRC / guardrail checks
├── duplicates.py     # check_duplicate + the embedding "pile" (Chroma) for dup detection
├── llm.py            # OpenAI client wrapper — chat() + embeddings, key passed in
├── byok.py           # Key resolution, format validation, redaction (no os.environ writes)
├── fetcher.py        # fetched_repo() — .py-only GitHub API fetch into a temp dir (shallow-clone fallback)
├── dashboard.py      # Streamlit single-page live review (ephemeral, GitHub-link only)
├── .streamlit/
│   └── config.toml   # deployment hardening — telemetry off, XSRF/CORS on, no tracebacks on the public page
├── requirements.txt
└── .env.example
```

The intentionally-vulnerable **fixtures live in a separate repo**, [`code-review-sample`](https://github.com/sriram98669704/code-review-sample), which the dashboard pulls straight from GitHub at runtime — so it's never a nested git repo inside this project. It's **one example input, not something the tool is built around** — a compact suite whose files between them exercise every lens:

```
code-review-sample/
├── accounts.py   # PII in logs (compliance), mutable default arg (guardrail)
├── api.py        # SQL injection; exact cross-file dup of db.get_user; two SAME-FILE path traversals — one a helper neutralizes (triage overrules it) + one it doesn't (kept)
├── db.py         # hardcoded secret (top-level), SQL injection, in-file near-duplicate (get_user ↔ fetch_user), one clean control (user_exists)
├── downloads.py  # CROSS-FILE path-traversal handlers — fetch_report / fetch_avatar, each flagged in isolation; the real verdict lives two hops away in paths.py
└── paths.py      # the helpers downloads.py calls — strip_traversal (os.path.basename → safe, drop) vs collapse_slashes (only //→/, still vulnerable → keep)
```

---

## What it checks

| Lens | Examples it looks for |
|---|---|
| **Security** | hardcoded secrets, SQL injection, command injection, path traversal, insecure deserialization, `eval` of input, SSRF, missing authorization checks, disabled TLS verification, weak hashing |
| **Compliance (GRC)** | PII written to logs or files, secrets committed in source, sensitive-data handling |
| **Guardrails** | backdoor/hardcoded credentials, bare `except` / swallowed errors, mutable default arguments |
| **Duplicates** | redundant functions — both **exact copies** (byte-for-byte identical) and **near-duplicates** (functionally the same even when reworded — e.g. renamed variables or reordered lines), within a file *and* across files |

The [`code-review-sample`](https://github.com/sriram98669704/code-review-sample) repo is a deliberately small, intentionally-vulnerable suite that exercises every lens — including a cross-file duplicate — while staying cheap to scan. Its centerpiece is a **matched pair** of path-traversal cases that the per-function judge flags *identically*, because each builds a file path from user input via a helper:

- `read_upload` calls `normalize_path`, which only swaps slashes and leaves `../` intact → **genuinely vulnerable**, so triage **keeps** it.
- `read_export` calls `safe_name`, which is `os.path.basename` and strips `../` → **safe**, so triage **overrules** the finding and **drops** it, citing that line.

The same matched pair then repeats **across files and two hops deep** (`downloads.py` → `paths.py`), which is what actually exercises the interprocedural graph rather than a single direct call:

- `fetch_report` → `build_safe_path` → `strip_traversal` (`os.path.basename`, in another file) → **safe**, so triage **drops** it. A one-hop look at `build_safe_path` can't tell — it just returns `"reports/" + cleaned`; only following the chain into `strip_traversal` reveals the fix.
- `fetch_avatar` → `to_relative` → `collapse_slashes` (only `//`→`/`, leaves `../`) → still **vulnerable**, so triage **keeps** it.

These two have an identical call shape but the opposite verdict, decided only by reading a helper one file and one hop further down — so the suite tests the cross-file resolver and the taint-guided walk end to end, not just same-file calls. Telling any of these apart is impossible from the call site — it requires reading the helper and reasoning about whether it sanitises, which is exactly what the agent + triage do. Each file also contains at least one *safe* function, so the suite tests that the agent doesn't cry wolf. The fixtures carry **no comments hinting at the flaws**, so a finding means the agent genuinely *detected* it — not that it read the answer off a comment.

---

## Limitations

This is an **LLM-assisted** review of **Python only**. It reasons about code the way a reviewer would, which is its strength, but that also means:

- It **complements** pattern-based tools like bandit/semgrep — it doesn't replace them. Use both.
- **Finding counts may vary slightly between runs** — the model is not perfectly deterministic.
- It reviews code you point it at; a live run is **ephemeral** and nothing is persisted, so there's no run history to compare over time.
- The **agent loop is bounded by generous safety caps** — at most `MAX_FUNCTIONS` (200) functions reviewed per run and `MAX_ROUNDS` loop turns, both far above any normal review (the sample is ~30 functions). When a cap is hit the report **says what was skipped** rather than truncating silently.

### What the cross-file resolver won't follow

To read a helper in another file, the resolver has to *prove* which function a call points to, using only the imports and definitions it can see statically. When it can't prove it, it returns **nothing and keeps the finding** — it never guesses, because a wrong guess could read the wrong file's helper and wrongly **drop a real bug**. So it deliberately gives up (and keeps the finding) on:

- **A bare name defined in two files**, when the caller neither defines nor explicitly imports it — e.g. a helper whose name happens to exist in two different modules. Rare in clean code: a fallback, not a common path, since a real bare call is normally either defined in the same file or imported by name (both of which *do* resolve).
- **A method on a value whose type we don't track** — `items.save()` where `items` is a parameter or local.
- **A local variable that shadows a module name** — `db = connect(); db.run()` is never mistaken for `db.py`.
- **Star imports** — `from x import *; mystery()` hides where `mystery` came from.
- **Package-relative imports** — `from . import helper`.
- **Computed call targets** — `handlers[key]()`, `make_fn()()` (no name to resolve).

Third-party and stdlib calls (`requests.get`, `os.path.basename`) aren't repo functions, so the walk stops there and records them as leaf **sinks** — it can't see inside library code. In every one of these the finding is **kept, not dropped**: a little extra noise is the safe direction; silently dropping a real vulnerability is the one outcome a security tool must avoid.

### How deep the walk goes

The taint-guided call-graph walk is bounded on purpose: **3 hops deep, 12 helpers wide** per finding, with a visited-set cycle guard so it always terminates (no infinite loops, even on mutually-recursive code). Both bounds are **configurable** — `max_depth` / `max_helpers` on [`gather_evidence()`](callgraph.py) — not hardcoded magic numbers. A fix buried deeper than 3 hops, or past the 12th helper, won't be seen — and again the finding is **kept**. The taint tracking is **conservative**: it over-includes helpers rather than risk missing a tainted path, so a bundle can carry a helper or two that turn out irrelevant.

The walk seeds taint from a function's **parameters** *and* from a short list of known input **sources** — `input()`, `os.environ`/`os.getenv`, `sys.argv`, and Flask's `request.form`/`request.args`/`request.get_json()` — so a handler that takes **no parameters** but reads its input from one of those (a Flask view that reads `request.form` and passes it to a helper) still gets its chain walked and its finding cleared. The one *global* it seeds is Flask's `request`, and only its **caller-controlled members** — a known framework input boundary, not an arbitrary value. What it deliberately does **not** treat as a source is an **arbitrary module global**: a global can hold anything, so tainting every global read would over-drop — the unsafe direction. A finding like that is **kept**, and only the **agent** (whose `read_function` isn't taint-gated) may choose to open its helper.

### Rule-bounded, by stage

- **The judge only flags what a policy rule describes** — it is not a general bug finder. An issue with no matching rule in `policies/*.json` won't be reported. Each chunk is shown the **5 nearest rules per category**, and a grounding guard discards any finding that cites a rule it wasn't shown.
- **Triage only drops a finding when the model can cite the exact neutralising line** in the helpers it was handed. A real fix the resolver/walk never surfaced can't be cited — so it stays kept.
- **Duplicate detection reports the single nearest match** above `0.80` similarity per function, not every pair in a cluster; a near-duplicate just under that line won't be confirmed.

---

## What's next

- **Real-repo demo** — showcase runs against known-vulnerable projects: vulpy, DVPWA, and OWASP PyGoat. (Flask `request` is now a taint source, so a view that reads `request.form`/`request.args` and passes it to a helper has its cross-function chain walked — the case these repos exercise.)
- **Offline tests** — every pure, no-API surface ships with a free suite under `tests/` (run any with `python tests/test_<name>.py`): the import-aware resolver, the taint-guided call-graph walk, the agent-exploration summary, BYOK key resolution + redaction, the AST chunker, and the deterministic security checks. What's left to cover needs a key — integration tests of the LLM-dependent passes (judge, triage, duplicate-confirm) end to end.
- **Per-finding budget** — the per-run caps (`MAX_FUNCTIONS`, `MAX_ROUNDS`) now bound a whole review; a finer-grained token/tool-call budget *per finding* could bound how much investigation any single verdict can spend on a pathological chain.
- **Evals — measuring precision/recall.** A labelled corpus of known-vulnerable and known-safe functions, scored every run, so a prompt or model change can be *proven* to help rather than eyeballed. This is the biggest real gap: correctness today is read off the output, not measured.
- **Observability — structured run traces.** Per-run timing, token cost, and tool-call counts emitted as structured logs, so a slow or expensive run can be diagnosed afterward — a live run is ephemeral today, so there's nothing to inspect once it's gone.
- **Concurrency / scale.** Functions are judged sequentially — one `judge` call at a time, fine for a ~30-function demo but linear in repo size. Batching or parallelising the per-function calls within rate limits is what a large-repo run would need.
