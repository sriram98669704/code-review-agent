"""Streamlit UI for the code-review agent — a single live-review page.

Set your OpenAI key (env-first via .env, else paste it into the BYOK panel —
held in this browser session's RAM only, never written to os.environ, disk, or
logs), paste a public GitHub repo link, hit Run, and watch the agent's
decide → act → observe loop stream live inside st.status — it indexes the repo,
reviews every function (against the rules and for duplicate code), pulls in a
helper when a verdict depends on it, runs a triage pass that overrules any false
positive a helper proves safe, then writes the verdict.
The repo's .py files are fetched via the GitHub API into a temp dir (fetcher.py),
scanned, then deleted — local and deployed runs behave identically.

A live run is ephemeral: its result is shown on screen for the session and is
never saved to disk — exactly like llm-eval-lab's Live Test tab.
"""

import os
from html import escape

import streamlit as st

import inspect

from agent import run_agent, HONESTY_LINE
from byok import key_from_env, resolve_key, validate_key_format, redact
from callgraph import gather_evidence as _gather_evidence
from fetcher import parse_repo_url, fetched_repo

# Read the engine's real chain bounds straight from gather_evidence's defaults (triage calls
# it with those defaults), so the call-chain explanation can never drift from the depth/width
# the triage pass actually uses. Change the signature in callgraph.py and this text follows.
_chain_params = inspect.signature(_gather_evidence).parameters
CHAIN_MAX_DEPTH = _chain_params["max_depth"].default
CHAIN_MAX_HELPERS = _chain_params["max_helpers"].default

# ── Finding cards (mirrors llm-eval-lab's .finding-card styling) ─────────────
_SEVERITY_COLOR = {"high": "#dc2626", "medium": "#d97706", "low": "#65a30d"}

_CARD_CSS = """
<style>
.finding-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-left: 4px solid var(--sev-color, #888);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.finding-card .fc-top {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.55);
    margin-bottom: 4px;
}
.finding-card .fc-title {
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 4px;
}
.finding-card .fc-desc {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.78);
}
.finding-card .fc-fix-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #34d399;
    margin-top: 8px;
    margin-bottom: 4px;
}
.finding-card .fc-fix {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.8rem;
    white-space: pre-wrap;
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 6px;
    padding: 8px 10px;
    margin: 0;
    color: rgba(255,255,255,0.90);
    overflow-x: auto;
}
</style>
"""

# ── Severity bar chart — a tiny custom-HTML horizontal bar, coloured with the
#    same red/orange/green palette as the cards (st.bar_chart can't colour bars
#    per-category, and a self-contained bar adds no charting dependency).
_BAR_CSS = """
<style>
.sev-row {
    display: grid;
    grid-template-columns: 88px 1fr 32px;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.sev-label { font-size: 0.85rem; color: rgba(255,255,255,0.80); }
.sev-track {
    background: rgba(255,255,255,0.06);
    border-radius: 6px;
    height: 18px;
    overflow: hidden;
}
.sev-fill { height: 100%; border-radius: 6px; }
.sev-count { font-size: 0.85rem; text-align: right; color: rgba(255,255,255,0.80); }
</style>
"""


def render_severity_summary(findings):
    """Severity breakdown as coloured horizontal bars — a one-glance read on how
    bad the code is. Each bar's width is scaled to the largest count, so the most
    common severity fills its track and the rest are proportional to it. Returns
    early (draws nothing) when no finding carries a known severity."""
    order = [("high", "🔴 High"), ("medium", "🟠 Medium"), ("low", "🟢 Low")]
    counts = {sev: 0 for sev, _ in order}
    for f in findings:
        sev = f.get("severity")
        if sev in counts:
            counts[sev] += 1
    top = max(counts.values())
    if top == 0:
        return
    st.markdown(_BAR_CSS, unsafe_allow_html=True)
    for sev, label in order:
        n = counts[sev]
        pct = (n / top) * 100
        color = _SEVERITY_COLOR[sev]
        st.markdown(
            f'<div class="sev-row">'
            f'<div class="sev-label">{label}</div>'
            f'<div class="sev-track"><div class="sev-fill" '
            f'style="width:{pct:.0f}%;background:{color}"></div></div>'
            f'<div class="sev-count">{n}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Run-log tree — turns the flat list of on_step strings into a nested
#    repo → file → function tree, so progress reads as structure, not a wall of
#    identical bullets. Function chunks render solid; module-level chunks
#    (imports, top-level assignments — shown as <Import>, <Assign>) render dim.
_TREE_CSS = """
<style>
.runtree { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.82rem; line-height: 1.55; }
.runtree .rt-root { color: rgba(255,255,255,0.92); font-weight: 600; margin-bottom: 2px; }
.runtree .rt-file { color: rgba(255,255,255,0.95); font-weight: 600;
    margin: 10px 0 2px 14px; }
.runtree .rt-node { color: rgba(255,255,255,0.82); margin-left: 40px;
    position: relative; }
.runtree .rt-node::before { content: "├─"; position: absolute; left: -18px;
    color: rgba(255,255,255,0.30); }
.runtree .rt-node.mod { color: rgba(255,255,255,0.50); font-style: italic; }
.runtree .rt-name { color: #e5e7eb; }
.runtree .rt-desc { color: rgba(255,255,255,0.38); font-size: 0.76rem; }
.runtree .rt-done { color: #22c55e; margin: 10px 0 0 14px; font-weight: 600; }
</style>
"""


def _run_log_html(steps, repo_label=None):
    """Build the repo → file → function tree HTML from the on_step messages.

    Each file lists the functions we scanned (with a plain note of what we did to
    each), plus — if the file has any — ONE line for the code outside functions
    (imports and top-level constants like API_KEY = "...", which we also scan).
    The raw AST node tags (<Import>, <Assign>, …) never reach the screen.

    `repo_label` overrides the tree's root name: the agent calls index_repo
    itself with the temp clone path, so render_agent_log passes the clean
    `owner/repo` label here instead of showing that throwaway temp dir."""
    repo = repo_label
    files = []  # [(filename, [node_name, ...]), ...]
    for s in steps:
        if s.startswith("walking repo: ") and repo_label is None:
            repo = s[len("walking repo: "):]
        elif s.startswith("scanning file: "):
            files.append((s[len("scanning file: "):], []))
        elif s.startswith("mapping "):         # one per function chunk
            name = s[len("mapping "):]
            if files:
                files[-1][1].append(name)
        elif s.startswith("checking top-level code in "):   # one per file with top-level code
            if files and "<top-level>" not in files[-1][1]:
                files[-1][1].append("<top-level>")

    def node(name_html, desc, mod=False):
        cls = "rt-node mod" if mod else "rt-node"
        return (f'<div class="{cls}">{name_html} '
                f'<span class="rt-desc">— {desc}</span></div>')

    html = [_TREE_CSS, '<div class="runtree">']
    if repo:
        html.append(f'<div class="rt-root">📦 {escape(repo)}</div>')
    for fname, nodes in files:
        html.append(f'<div class="rt-file">📄 {escape(fname)}</div>')
        funcs = [n for n in nodes if not n.startswith("<")]
        has_top_level = any(n.startswith("<") for n in nodes)
        if has_top_level:
            # Everything outside a function — collapsed into one honest line.
            html.append(node("top-level code (imports & constants)",
                             "checked for risks", mod=True))
        for fn in funcs:
            html.append(node(f'<span class="rt-name">{escape(fn)}()</span>',
                            "mapped"))
    if any(s == "done" for s in steps):
        html.append('<div class="rt-done">✅ scan complete</div>')
    html.append("</div>")
    return "\n".join(html)


# ── Agent timeline — the same scan tree, wrapped in the agent's decide → act →
#    observe milestones, so Agent review reads as a loop you can watch, not just
#    a file walk. Reuses _run_log_html for the scan portion.
_AGENTLOG_CSS = """
<style>
.agentlog { font-size: 0.86rem; line-height: 1.5; }
.agentlog .al-head { font-weight: 600; color: rgba(255,255,255,0.92); margin-bottom: 8px; }
.agentlog .al-step { color: rgba(255,255,255,0.85); margin: 18px 0 0 0;
    border-left: 2px solid rgba(99,102,241,0.55);
    border-top: 1px solid rgba(255,255,255,0.10);
    padding: 18px 0 8px 14px; }
.agentlog .al-step code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    background: rgba(99,102,241,0.15); padding: 1px 5px; border-radius: 4px; color: #c7d2fe; }
.agentlog .al-pending { color: rgba(255,255,255,0.50); font-style: italic; }
.agentlog .al-tree { margin: 8px 0 6px 6px; }
.agentlog .al-list { margin: 6px 0 6px 2px; padding-left: 18px; }
.agentlog .al-list li { margin: 3px 0; }
.agentlog .al-arrow { color: rgba(255,255,255,0.55); }
.agentlog .al-file { color: rgba(255,255,255,0.5); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.agentlog .al-foot { margin-top: 26px; padding-top: 14px; color: rgba(255,255,255,0.72);
    border-top: 1px solid rgba(255,255,255,0.12); }
.agentlog .al-num { color: #fbbf24; font-weight: 700; }
.agentlog .al-chain { margin: 12px 0 4px 2px; }
.agentlog .al-find { background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.30); border-radius: 8px;
    padding: 10px 14px; margin: 12px 0; }
.agentlog .al-node { padding: 1px 0; color: rgba(255,255,255,0.92); }
.agentlog .al-node code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    background: rgba(99,102,241,0.15); padding: 1px 6px; border-radius: 4px; color: #c7d2fe; }
.agentlog .al-hop { color: rgba(255,255,255,0.40); padding-left: 6px; font-size: 0.82rem; }
.agentlog .al-tag { color: rgba(255,255,255,0.50); font-weight: 400; font-size: 0.82rem; }
.agentlog .al-verdict-drop { color: #6ee7a8; font-weight: 600; margin-top: 8px;
    padding-top: 6px; border-top: 1px dashed rgba(255,255,255,0.12); }
.agentlog .al-verdict-keep { color: #fca5a5; font-weight: 600; margin-top: 8px;
    padding-top: 6px; border-top: 1px dashed rgba(255,255,255,0.12); }
.agentlog .al-verdict-drop code, .agentlog .al-verdict-keep code {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    background: rgba(16,185,129,0.14); padding: 1px 6px; border-radius: 4px; color: #d1fae5; }
.agentlog .al-note { margin-top: 12px; color: rgba(255,255,255,0.80); }
.agentlog .al-explore { border-left-color: rgba(251,191,36,0.55); }
</style>
"""


def _split_loc(loc):
    """'owner/repo/api.py::get_user' -> ('api.py', 'get_user'). With no file part,
    the file is '' and the whole string is the name."""
    loc = (loc or "").strip()
    if "::" in loc:
        path, name = loc.rsplit("::", 1)
        return path.rsplit("/", 1)[-1], name
    return "", loc


def _fmt_loc(loc):
    """Render a 'path::name' location as 'file → name()' (just 'name()' if no file)."""
    fname, name = _split_loc(loc)
    code = f'<code>{escape(name)}()</code>'
    return f'<span class="al-file">{escape(fname)}</span> → {code}' if fname else code


def _parse_triage_step(t):
    """Split 'path::name (rule) via helper1, helper2 -> sink1, sink2' into
    (loc, rule, helpers, sinks):
      loc     - the finding's 'path::name';
      rule    - its rule id (e.g. 'path-traversal');
      helpers - the 'path::name' of each function on the tainted path the agent read;
      sinks   - the external calls (open(), os.path.basename(), ...) those reach.
    The ' -> ' separates helpers from sinks; it is NOT a per-helper arrow, so the sink
    list is never glued onto the last helper (the old bug). Either side may be empty."""
    head, via = t, ""
    if " via " in t:
        head, via = t.split(" via ", 1)
    loc, rule = head, ""
    if head.endswith(")") and " (" in head:
        loc, rest = head.split(" (", 1)
        rule = rest[:-1]
    helpers_part, sinks_part = via, ""
    if " -> " in via:
        helpers_part, sinks_part = via.split(" -> ", 1)
    helpers = [h.strip() for h in helpers_part.split(",") if h.strip()]
    sinks = [s.strip() for s in sinks_part.split(",") if s.strip()]
    return loc.strip(), rule, helpers, sinks


def _strip_report_title(md):
    """The model is told not to title its report, but if it still opens with a bare
    'Final Risk Report'/'Risk Report' line (optionally a markdown heading), drop that
    one line so it doesn't stack awkwardly under our own subheader."""
    lines = (md or "").lstrip().split("\n")
    if lines:
        first = lines[0].lstrip("#").strip().rstrip(":").lower()
        if first in {"final risk report", "risk report", "risk verdict",
                     "final report", "agent's risk verdict"}:
            return "\n".join(lines[1:]).lstrip()
    return md


def _drop_match(loc, dropped):
    """The dropped-finding dict (carrying the exact neutralising line) for a triaged
    'path::name', matched on function name + file. None means this finding was KEPT."""
    fname, name = _split_loc(loc)
    for d in (dropped or []):
        if d.get("name") == name and str(d.get("path", "")).rsplit("/", 1)[-1] == fname:
            return d
    return None


def _chain_cards_html(triaged, dropped):
    """One vertical call-graph card per triaged finding, read top-to-bottom: the flagged
    function, the function(s) its tainted input flows into, the external sink where the
    risk would land, then the verdict. The step stream gives a FLAT set of helpers (not
    their parent-child structure), so we list them as siblings rather than invent a
    helper->helper chain. Dropped findings show the exact safe line; kept findings say
    nothing on the path removed the risk. Cross-file hops are marked '(another file)'."""
    cards = []
    for t in triaged:
        loc, rule, helpers, sinks = _parse_triage_step(t)
        ffile, fname = _split_loc(loc)
        rule_txt = escape(rule.replace("-", " ")) if rule else "risk"
        rows = [f'<div class="al-node">🚩 <code>{escape(fname)}()</code>'
                f'<span class="al-tag"> &nbsp;{escape(ffile)} · flagged: {rule_txt}</span></div>']
        if helpers:
            lead = ("its input flows into:" if len(helpers) == 1
                    else "its input flows into these function(s):")
            rows.append(f'<div class="al-hop">&nbsp;&nbsp;│ {lead}</div>')
            for h in helpers:
                hfile, hname = _split_loc(h)
                hname = hname[:-2] if hname.endswith("()") else hname
                tag = (hfile + " (another file)") if hfile and hfile != ffile else hfile
                tag_html = f'<span class="al-tag"> &nbsp;{escape(tag)}</span>' if tag else ""
                rows.append(f'<div class="al-node">&nbsp;&nbsp;↳ <code>{escape(hname)}()</code>{tag_html}</div>')
        if sinks:
            verb = ("which reaches:" if (len(sinks) == 1 and helpers)
                    else "which reach:" if helpers else "reaches:")
            rows.append(f'<div class="al-hop">&nbsp;&nbsp;│ {verb}</div>')
            for s in sinks:
                sname = s[:-2] if s.endswith("()") else s
                rows.append('<div class="al-node">&nbsp;&nbsp;⚠️ '
                            f'<code>{escape(sname)}()</code>'
                            '<span class="al-tag"> &nbsp;external call — where the risk would land</span></div>')
        d = _drop_match(loc, dropped)
        if d:
            cite = escape(((d.get("triage") or {}).get("line") or "").strip())
            cite_html = f'<code>{cite}</code> ' if cite else ""
            rows.append(f'<div class="al-verdict-drop">✅ {cite_html}removes the risk &nbsp;→&nbsp; '
                        f'<b>DROPPED</b> as a false alarm</div>')
        else:
            rows.append('<div class="al-verdict-keep">❌ nothing on this path removes the risk '
                        '&nbsp;→&nbsp; <b>KEPT</b> as a real risk'
                        '<span class="al-tag"> &nbsp;(when unsure it keeps — dropping a real bug '
                        'is the one mistake we never want)</span></div>')
        cards.append(f'<div class="al-find">{"".join(rows)}</div>')
    return "".join(cards)


# ── Agentic-exploration block — the helpers the agent opened on its OWN (read_function),
#    rendered ABOVE the triage block. It stands on its own: it reports WHAT the agent chose
#    to do while reviewing, in plain language, and makes no claim about the triage pass (the
#    taint / depth / width mechanics are introduced later, in the triage block itself).
_EXPLORE_INTRO = (
    "A finding’s risk often depends on a helper the flagged function calls. On its own "
    "initiative, the agent opened these helpers’ source while reviewing — to weigh them "
    "itself (the <i>act → observe</i> part of the loop). Here’s each one, and which "
    "function it opened it for."
)


def _exploration_block_html(exploration):
    """The agent's own read_function exploration as one timeline step. Only helpers the
    agent opened WHILE reviewing a specific flagged function are shown (entry is not None) —
    the 'while reviewing X(), opened Y()' framing needs that X. Helpers the agent opened off
    any flagged function (entry is None) are dropped: there's no function to frame them
    against. The count is computed from what's shown, never hardcoded. This block stands on
    its own — it reports what the agent chose to do and makes no claim about the triage pass."""
    opened = [o for o in (exploration or {}).get("opened", []) if o.get("entry")]
    if not opened:                                  # empty state, stated plainly
        return (
            '<div class="al-step al-explore">🕵️ <b>The agent didn’t open any helper on '
            'its own this run</b><br><br>'
            'While reviewing, the agent <i>can</i> pull in a helper’s source to settle a '
            'verdict that hinges on code it can’t see — but whether it does, and how far '
            'it digs, is the <b>model’s own call</b>. This run it opened none.</div>')
    n = len(opened)
    rows = []
    for o in opened:
        entry = escape(str(o["entry"]))
        efile = escape(str(o.get("entry_file") or ""))
        helper = escape(str(o.get("helper", "")))
        hfile = escape(str(o.get("helper_file") or ""))
        entry_lbl = f'{efile}’s ' if efile else ""
        helper_lbl = f'{hfile}’s ' if hfile else ""
        rows.append(
            f'<div class="al-node">📂 While reviewing {entry_lbl}<code>{entry}()</code>, '
            f'the agent opened its helper {helper_lbl}<code>{helper}()</code></div>')
    return (
        f'<div class="al-step al-explore">🕵️ <b>While reviewing, the agent opened '
        f'<span class="al-num">{n}</span> helper(s) on its own</b><br><br>'
        f'{_EXPLORE_INTRO}'
        f'<div class="al-chain">{"".join(rows)}</div>'
        '<div class="al-note">→ These are the agent’s own choices, shown for '
        'transparency.</div></div>')


# Precise per-reason wording for the helpers that sit ON a flagged chain but that triage's
# guaranteed walk did NOT reach — shown in the triage block's "didn't reach" note, named
# against the real bounds. {entry}/{helper} and their {*_file} are filled from the run.
_REASON_DETAIL = {
    "no-seed": ("{entry_file}’s <code>{entry}()</code> takes no parameters and reads its "
                "input from a module-level global, which this walk deliberately doesn’t "
                "treat as a tracked source — so it has no thread to follow and never starts "
                "a walk there. Only the agent reached {helper_file}’s <code>{helper}()</code>."),
    "depth": ("{helper_file}’s <code>{helper}()</code> sits more than {depth} hops down "
              "{entry_file}’s <code>{entry}()</code>’s chain — past the depth limit above — "
              "so the walk stops before it."),
    "width": ("triage had already gathered its {width}-helper limit on {entry_file}’s "
              "<code>{entry}()</code>’s chain before it reached {helper_file}’s "
              "<code>{helper}()</code>."),
    "off-path": ("the tracked input never flows into {helper_file}’s <code>{helper}()</code>, "
                 "so the taint-guided walk prunes it from {entry_file}’s "
                 "<code>{entry}()</code>’s chain."),
    "unresolvable": ("reaching {helper_file}’s <code>{helper}()</code> means passing a call "
                     "triage can’t pin to one definition, so it stops short."),
}


def _beyond_reason_lines(exploration):
    """Data-driven note, for the triage block, of helpers that sit ON a flagged chain but
    that triage's guaranteed walk did NOT reach — each tagged with WHY, named against the
    real bounds. Helpers the agent opened off any flagged chain (entry is None) are
    excluded: they were never on a chain triage walks, so 'didn't reach' wouldn't apply.
    Empty string when there's nothing on-chain that triage missed."""
    beyond = [o for o in (exploration or {}).get("opened", [])
              if o.get("beyond_forced") and o.get("entry")]
    if not beyond:
        return ""
    items = []
    for o in beyond:
        tmpl = _REASON_DETAIL.get(o.get("reason"), _REASON_DETAIL["off-path"])
        items.append('<li>' + tmpl.format(
            entry=escape(str(o["entry"])),
            entry_file=escape(str(o.get("entry_file") or "")),
            helper=escape(str(o.get("helper", ""))),
            helper_file=escape(str(o.get("helper_file") or "")),
            depth=CHAIN_MAX_DEPTH, width=CHAIN_MAX_HELPERS) + '</li>')
    return ('<br><br><b>A note on what triage’s walk didn’t reach</b> (this run): '
            '<ul class="al-list">' + "".join(items) + '</ul>')


def render_agent_log(steps, done=False, repo_label=None, dropped=None, exploration=None):
    """Render run_agent's on_step stream as a plain, step-by-step account of what the
    agent did, in order:

      1. Indexed the repo       — map every function, settle the top-level code.
      2. Reviewed each one       — rules AND duplicates, together, per function.
      3. Followed the call chain — for findings whose safety depends on a function
                                   they call, walk the tainted path to the real code
                                   and drop the ones a line provably makes safe.
      4. Wrote the verdict       — the report below.

    Each step renders as soon as its on_step messages appear, so the run reads as a
    live sequence anyone can follow. `done` adds the closing lines (only true once
    the run ends)."""
    indexed = any(s.startswith("agent calling index_repo") for s in steps)
    scan_done = any(s == "done" for s in steps)        # the index walk has finished
    # one "reviewing path::name" per function definition, kept in order (so the live
    # counter can name the function currently under review and the count matches the tree).
    review_locs = list(dict.fromkeys(s[len("reviewing "):]
                       for s in steps if s.startswith("reviewing ")))
    # Total functions to review = one "mapping <fn>" per function from the index walk, which
    # has fully finished before review starts. Guard so the live "N of total" can never read
    # past the total if the two ever drift (e.g. 17 of 16).
    review_total = max(sum(1 for s in steps if s.startswith("mapping ")), len(review_locs))
    triaged = [s[len("triaging "):] for s in steps if s.startswith("triaging ")]

    html = [_AGENTLOG_CSS, '<div class="agentlog">',
            '<div class="al-head">🤖 What the agent did, step by step</div>']

    # 1) INDEX — the structural map + settle the top-level code (no judging yet).
    # Tense tracks progress: "Indexing…" while the walk runs, "Indexed" once done.
    if not indexed:
        html.append('<div class="al-step al-pending">🧠 Deciding what to do…</div>')
    else:
        cls = "al-step" if scan_done else "al-step al-pending"
        verb = "Indexed" if scan_done else "Indexing"
        tail = "" if scan_done else "…"
        html.append(f'<div class="{cls}">🗂️ <b>{verb} the repo</b>{tail} — walking every file and '
                    'noting where each function lives. This step only <i>locates</i> the functions; '
                    'each one gets reviewed in the next step. Code outside any function (imports, '
                    'constants like API keys) is <b>also checked for risks right here</b> — against '
                    'the same rules as everything else, but because it has no helper the agent could '
                    'open and dig into, a single pass settles it:</div>')
        if any(s.startswith("walking repo: ") for s in steps):
            html.append('<div class="al-tree">'
                        + _run_log_html(steps, repo_label=repo_label) + '</div>')

    # 2) REVIEW — rules AND duplicates, together, for every function. While the run is
    # still going the count is climbing, so it reads "Reviewing … (N)…"; once done it
    # settles to the past tense "Reviewed … (N)".
    if review_locs:
        # Review is finished once triage has begun (or the whole run is done) — so the line
        # flips to past-tense "Reviewed all N" the moment the call chain starts, instead of
        # staying stuck on "Reviewing… N of N" while triage runs.
        review_done = done or bool(triaged)
        cls = "al-step" if review_done else "al-step al-pending"
        # WHAT one pass covers — "one pass EACH" (per function), never "all in one pass"
        # (which would wrongly imply all functions in a single shot). Two things differ by state:
        # tense (running -> "is", finished -> "was"), and the duplicate-check tail (mid-run a
        # function can only be compared with the ones already done -> "seen so far"; once
        # finished, every function has been compared with every other -> "the others").
        detail_body = ('checked for <b>security vulnerabilities</b>, compliance '
                       '(GRC) issues and guardrail problems, <i>and</i> compared against ')
        if review_done:                         # finished: past tense, every function vs every other
            detail = 'each function was ' + detail_body + 'the others to catch duplicate code'
            html.append(f'<div class="{cls}">⚖️ <b>Reviewed all {len(review_locs)} functions '
                        f'— one pass each.</b> In that single pass, {detail}.</div>')
        else:                                   # running: present tense, only the ones reviewed so far
            detail = 'each function is ' + detail_body + 'the functions seen so far to catch duplicate code'
            html.append(f'<div class="{cls}">⚖️ <b>Reviewing every function — one pass each</b>… '
                        f'&nbsp;<span class="al-arrow">now:</span> {_fmt_loc(review_locs[-1])} '
                        f'&nbsp;<span class="al-arrow">({len(review_locs)} of {review_total})</span><br>'
                        f'In that single pass, {detail}.</div>')
    elif scan_done and not done:
        html.append('<div class="al-step al-pending">⚖️ Reviewing each function — '
                    'security, GRC, guardrails + duplicates…</div>')

    # 2.5) THE AGENT'S OWN EXPLORATION — the helpers it opened itself (read_function),
    # placed where it happened in the loop: during review, BEFORE the guaranteed triage
    # pass below. Only in the final render (done) and only when the engine handed us the
    # summary; live renders skip it (the full set of reads isn't known until the run ends).
    if done and exploration is not None:
        html.append(_exploration_block_html(exploration))

    # 3) FOLLOW THE CALL CHAIN — the single place "the agent read other functions" is
    # told, drawn as a graph: flagged function → the function(s) it calls → the sink,
    # with the verdict (dropped / kept) on each. Replaces the old, confusing split of
    # "looked closer" + "triage" and the 2-vs-4 mismatch, and names the taint / call
    # graph / depth limits in plain words so nobody assumes it's a one-level check.
    if triaged and done:
        n = len(triaged)
        total = len(review_locs)
        drop_n = len(dropped or [])
        keep_n = n - drop_n
        cards = _chain_cards_html(triaged, dropped)
        reason_lines = _beyond_reason_lines(exploration)  # data-driven "didn't reach" note
        html.append(
            f'<div class="al-step">🔗 <b>Of the <span class="al-num">{total}</span> functions '
            f'reviewed by the AI agent, it raised several findings. A process called triage '
            f'then reviewed all those findings and determined that '
            f'<span class="al-num">{n}</span> functions’ vulnerability assessment hinges on '
            f'other functions — followed via the call chain (up to <b>{CHAIN_MAX_DEPTH}</b> '
            f'deep and <b>{CHAIN_MAX_HELPERS}</b> helpers wide — both configurable)</b><br><br>'
            'After the agent reviews functions and raises findings, we run a separate process '
            'called <b>triage</b>. Triage builds a <b>call graph</b> and follows it along the '
            '<b>tainted path</b> for each finding — “taint” meaning untrusted input we track '
            'from function to function, and a line that <b>sanitises</b> it (cleans the '
            'untrusted value so it can’t do harm) <b>removes the taint</b>. This is the same '
            'technique tools like <b>CodeQL</b> and <b>Semgrep</b> use.'
            '<br><br>'
            'Not every finding gets a call graph, though — only the ones whose risk hinges on '
            'what a <i>called</i> function does with the input. A function with no tainted '
            'input passing through it (a hard-coded secret, or SQL built straight from a value '
            'that’s never handed off) doesn’t need one: the AI agent already flagged it '
            'independently, on its own code alone, and nothing downstream can change that '
            'verdict — so there’s no chain to build and no helper to disprove it with.'
            '<br><br>'
            'The walk itself has real limits, by design, always erring toward <b>keeping</b> a '
            'finding rather than risking a wrong drop. It only follows a call it can '
            '<b>statically prove</b> the target of — a name that’s ambiguous (defined in two '
            'files, reached through a star import, a computed call like '
            '<code>handlers[key]()</code>) is left unresolved and the finding stays. It stops '
            'at the edge of the repo — a call into a third-party or stdlib function '
            '(<code>requests.get</code>, <code>os.path.basename</code>) is recorded as a leaf '
            f'<b>sink</b>, not followed inside. It’s bounded on purpose — <b>{CHAIN_MAX_DEPTH} '
            f'hops deep, {CHAIN_MAX_HELPERS} helpers wide</b> per finding, with a visited-set '
            'guard so cycles (a→b→a) can’t loop forever — so a fix buried deeper than that '
            'won’t be seen, and again the finding is kept rather than guessed away. The same '
            'applies to the <b>seed</b>: the walk starts from a function’s parameters and a '
            'short list of known input sources (<code>input()</code>, <code>os.environ</code>, '
            '<code>sys.argv</code>); a function that takes no parameters and instead reads an '
            'arbitrary module-level global isn’t seeded, because treating every global as '
            'tainted would over-drop. In every one of these cases the finding is kept, not '
            'dropped — the only one who can still go open that helper anyway is the <b>agent '
            'itself</b>, on its own initiative, which isn’t bound by any of these limits.'
            '<br><br>'
            '<b>Why triage at all, separate from the agent?</b> The first review — the AI '
            'agent — is deliberately paranoid: it over-flags rather than risk missing a real '
            'bug. And the agent’s own exploration isn’t guaranteed — it won’t necessarily open '
            'every helper a finding depends on, or follow a chain to its full depth; how far '
            'it digs is the model’s own call, not a fixed rule. A model is also poor at '
            'overturning its own call: once it has flagged a function, that verdict sits in '
            'its context, and more often than not it fails to correct itself — even when the '
            'chain ends in a function that makes the input safe, it re-reads its own flag and '
            'leaves the finding standing. So triage — a <b>fresh reviewer</b>, separate from '
            'the agent that raised the finding — gets handed the finding plus the chain it '
            'gathers, with no stake in the original verdict and the opposite job: to '
            '<i>disprove</i> it. It may drop a finding <b>only</b> if it can quote the exact '
            'line that makes it safe — otherwise the finding stays.'
            '<br><br>'
            f'For these <span class="al-num">{n}</span>, the flagged function passes its '
            'inputs <i>on</i> to other functions, so whether it’s truly dangerous depends on '
            'what that <b>whole chain</b> does with them. (A <b>call chain</b> is just that '
            'hand-off: A calls B, which may call C, and so on — the input travels along it.)'
            f'{reason_lines}'
            '<br><br>'
            '<b>What the chain found</b> — read each card top-to-bottom: the flagged '
            'function, the function(s) it calls, the risky operation at the bottom, then '
            'the verdict.'
            f'<div class="al-chain">{cards}</div>'
            f'<div class="al-note">→ <span class="al-num">{drop_n}</span> dropped as false '
            f'alarm(s) — a line proved them safe · <span class="al-num">{keep_n}</span> kept '
            f'as real risk(s) — nothing on the path could disprove them. The dropped one(s), '
            f'with the exact safe line, are in the <b>“Overruled”</b> panel below.</div>'
            '</div>')
    elif triaged:                              # mid-run: verdicts not known yet
        html.append(f'<div class="al-step al-pending">🔗 Following the call chain — reading the '
                    f'function(s) that {len(triaged)} finding(s) depend on, down the tainted '
                    f'path, to see if any line removes the risk…</div>')
    elif done:
        html.append('<div class="al-step">🔗 <b>Followed the call chain</b> — no finding passed '
                    'its input into another function, so there was nothing deeper to check; each '
                    'verdict was settled in the single review pass above.</div>')

    # 4) WRITE — the verdict (rendered below the timeline).
    if done:
        html.append('<div class="al-step">📝 <b>Wrote the final risk verdict</b> — the '
                    'plain-English report further down the page.</div>')
        html.append('<div class="al-foot">⬇️ <b>See the full results below</b> — every finding '
                    'with a suggested fix, the false positive(s) triage overruled, the duplicate '
                    'functions, and the agent’s plain-English verdict.</div>')
    html.append("</div>")
    st.markdown("\n".join(html), unsafe_allow_html=True)


def _fix_code_html(fix):
    """Escape a fix snippet for the finding card. Streamlit runs the card HTML
    through a markdown pass that collapses real newlines and leading spaces, so a
    multi-line fix would otherwise flatten onto one line (hiding any code after a
    trailing comment). Encode line breaks as <br> and indentation as &nbsp; so a
    multi-line fix keeps its shape; single-line fixes are unaffected."""
    out = []
    for line in fix.split("\n"):
        body = line.lstrip(" ")
        out.append("&nbsp;" * (len(line) - len(body)) + escape(body))
    return "<br>".join(out)


def render_findings(findings):
    if not findings:
        st.success("No findings — nothing matched a security, GRC, or guardrail rule.")
        return
    st.markdown(_CARD_CSS, unsafe_allow_html=True)
    st.caption(
        "Grouped by category; within each, most severe first. Each card shows "
        "**file · function · line · severity**, then the rule it broke and a "
        "plain-English explanation. The coloured left border marks severity — "
        "🔴 high · 🟠 medium · 🟢 low."
    )
    # Friendly category headings (the raw tags are terse: grc, guardrail, …).
    cat_label = {
        "security": "🔒 Security",
        "grc": "📋 Compliance (GRC — governance, risk & compliance)",
        "guardrail": "🛡️ Guardrails (code-safety practices)",
    }
    # Fixed category order (most security-critical first), then severity within.
    cat_order = {"security": 0, "grc": 1, "guardrail": 2}
    sev_rank = {"high": 0, "medium": 1, "low": 2}
    by_cat = {}
    for f in findings:
        by_cat.setdefault(f.get("category", "other"), []).append(f)
    for cat in sorted(by_cat, key=lambda c: cat_order.get(c, 99)):
        st.markdown(f"**{cat_label.get(cat, cat.title())}** ({len(by_cat[cat])})")
        for f in sorted(by_cat[cat], key=lambda x: sev_rank.get(x.get("severity"), 9)):
            color = _SEVERITY_COLOR.get(f.get("severity"), "#888")
            fix = (f.get("fix") or "").strip()
            fix_html = (
                '\n  <div class="fc-fix-label">✅ Suggested fix</div>'
                f'\n  <pre class="fc-fix">{_fix_code_html(fix)}</pre>'
                if fix else ""
            )
            st.markdown(
                f"""<div class="finding-card" style="--sev-color: {color}">
  <div class="fc-top">{f.get('path', '')} · {f.get('name', '')} · line {f.get('line', '?')} · {f.get('severity', '?').upper()}</div>
  <div class="fc-title">{f.get('rule_title', f.get('rule_id', ''))}</div>
  <div class="fc-desc">{f.get('explanation', '')}</div>{fix_html}
</div>""",
                unsafe_allow_html=True,
            )


def _qualify(name, path):
    """'get_user' + 'owner/repo/db.py' -> 'get_user <span>(db.py)</span>'. The file is
    muted so the function name still leads, but a reader always sees WHICH file each
    side of a duplicate lives in — essential when the same name exists in two files
    (the sample defines get_user in both db.py and api.py)."""
    fname = path.rsplit("/", 1)[-1] if path else ""
    tag = f' <span style="opacity:.6">({escape(fname)})</span>' if fname else ""
    return f"{escape(str(name))}{tag}"


def render_duplicates(duplicates):
    if not duplicates:
        st.success("No duplicates — no two functions were near-identical.")
        return
    st.markdown(_CARD_CSS, unsafe_allow_html=True)
    st.caption(
        "Functions the agent judged redundant — it catches **exact copies** "
        "(byte-for-byte identical) **and** near-duplicates: **different code that does "
        "the same job**, matched by *meaning* rather than text, so renamed variables or a "
        "reshuffled body don't slip past. It finds candidates by meaning (embedding "
        "similarity), then an **LLM confirms** each pair is a real duplicate before "
        "flagging. **Score** is that similarity from 0 to 1 — 1.00 means identical."
    )
    for d in duplicates:
        kind = d.get("kind", "similar")
        badge = "🟦 EXACT copy" if kind == "exact" else "🟪 SIMILAR logic"
        st.markdown(
            f"""<div class="finding-card" style="--sev-color: #2563eb">
  <div class="fc-top">{escape(d.get('path', ''))} · {badge} · score {d.get('score', 0):.2f}</div>
  <div class="fc-title">{_qualify(d.get('name', ''), d.get('path', ''))} duplicates {_qualify(d.get('duplicate_of', ''), d.get('duplicate_of_path', ''))}</div>
  <div class="fc-desc">{d.get('reason', '')}</div>
</div>""",
            unsafe_allow_html=True,
        )


def render_dropped(dropped, kept=None):
    """The override, shown plainly: findings the per-function judge raised but the
    triage pass KILLED as false positives — each with the helper and the exact line
    that neutralises the risk. This is the agent doing what a plain scanner can't:
    reading a helper and overruling a verdict on the evidence.

    `kept` is the surviving findings. When a dropped finding has a same-rule sibling
    that was KEPT (the sample's centerpiece: `read_export` is dropped while its near-
    twin `read_upload` is kept — same path-traversal rule, but only one helper truly
    sanitises), we name that sibling so the contrast is impossible to miss."""
    if not dropped:
        return
    kept = kept or []
    st.markdown(_CARD_CSS, unsafe_allow_html=True)
    st.caption(
        "**Flagged, then cancelled.** The first review flagged these as risky. The "
        "call-chain check then opened the function each one calls, followed the tainted "
        "input, and found the **exact line that removes the risk** — so they are not real. "
        "A plain pattern-scanner would wrongly keep all of these. Each card shows the path "
        "the agent walked and the line that settled it. Where the **same kind of risk was "
        "KEPT** in another function, we say so — that is the whole point: the agent tells "
        "real risks from safe ones by reading what each function calls, not by the pattern."
    )
    for d in dropped:
        t = d.get("triage", {})
        helper = escape(str(t.get("helper", "") or ""))
        name = escape(str(d.get("name", "") or ""))
        line = escape((t.get("line") or "").strip())
        reason = escape(str(t.get("reason", "") or ""))
        # The path the agent walked, drawn left-to-right: flagged function → the function
        # it calls → the exact line in that function that removes the risk.
        chain = (
            '\n  <div class="fc-desc" style="margin-top:10px;color:rgba(255,255,255,0.88)">'
            '<b>Path the agent walked:</b><br>&nbsp;&nbsp;🚩 <code>' + name + '()</code> '
            '<span style="color:rgba(255,255,255,0.45)">→ passes its input to →</span> '
            '<code>' + helper + '()</code> '
            '<span style="color:rgba(255,255,255,0.45)">→ which runs the safe line below</span>'
            '</div>'
            if helper else ""
        )
        evidence = (
            '\n  <div class="fc-fix-label" style="color:#6ee7a8">The exact line that makes it safe</div>'
            f'\n  <pre class="fc-fix">{line}</pre>'
            if line else ""
        )
        st.markdown(
            f"""<div class="finding-card" style="--sev-color: #6b7280">
  <div class="fc-top">{escape(str(d.get('path', '')))} · {escape(str(d.get('name', '')))} · line {d.get('line', '?')} · was {escape(str(d.get('severity', '?')).upper())}</div>
  <div class="fc-title">❌ {escape(str(d.get('rule_title', d.get('rule_id', ''))))} — ruled SAFE (false alarm): it calls <code>{helper}()</code>, which removes the risk</div>
  <div class="fc-desc">{reason}</div>{chain}{evidence}
</div>""",
            unsafe_allow_html=True,
        )

    # The contrast, stated ONCE for the whole panel (not repeated under each card): the
    # findings under the SAME rule that were KEPT as real — the "we read the code, not the
    # pattern" headline. Same rule as a dropped one, not itself dropped, still surviving.
    dropped_rules = {d.get("rule_id") for d in dropped}
    dropped_names = {d.get("name") for d in dropped}
    twin_names = sorted({str(f.get("name")) for f in kept
                         if f.get("rule_id") in dropped_rules
                         and f.get("name") not in dropped_names})
    if twin_names:
        names = ", ".join(f"`{n}()`" for n in twin_names)
        st.markdown(
            f"↔ **The same kind of risk was KEPT as a real bug in:** {names} — see "
            "**Findings** above. Those pass their input to a function that does **not** "
            "sanitise it, so they stay flagged. The agent told them apart from the dropped "
            "ones only by reading what each function calls — not by the pattern; a plain "
            "scanner would treat all of them identically."
        )


st.set_page_config(page_title="Code Review Agent", page_icon="🔍", layout="wide")

st.title("🔍 Code Review Agent")
st.caption(
    "An LLM agent that reviews Python code the way a human reviewer would: it reads "
    "each function and flags security holes, compliance (GRC) issues, weak guardrails, "
    "and duplicated logic — then explains every finding in plain English. When a flagged "
    "risk is actually neutralised by another function it calls, a separate, skeptical triage "
    "pass reads that function and **overrules the false positive** — so you see real risks, "
    "not noise. It "
    "runs live against the code you point it at; nothing is saved — each run is shown on "
    "screen only."
)

st.info(
    "**Each review makes live API calls.**  \n"
    "A run costs a few cents depending on how many functions are in the target "
    "(one embedding per function, plus one judge call per function that matches "
    "a rule).  \n"
    "A pasted key is held **in memory for this browser session only** — never "
    "written to disk, never logged, gone when you close the tab."
)

# A run is in progress only during the execute pass (PASS 2). While it runs,
# every control is disabled so inputs can't change mid-flight — the canonical
# Streamlit "freeze the form while busy", same as llm-eval-lab's Live Test.
running = st.session_state.get("review_running", False)

# ── BYOK key panel — env-first, then per-session pasted key ──────────────────
# Pasted key lives ONLY in st.session_state (per-session server RAM) — never
# written to os.environ, disk, or logs. The widget key carries a generation
# number so "Clear key" can fully reset the box by bumping the generation, the
# same robust reset llm-eval-lab uses for its keys.
_env_key = key_from_env(os.environ)

if _env_key:
    st.caption(
        "🔑 Using the OPENAI_API_KEY from your local environment — sent only "
        "to OpenAI, never stored or logged."
    )
else:
    _gen = st.session_state.get("byok_gen", 0)
    _wkey = f"byok_input_{_gen}"
    _saved = (st.session_state.get(_wkey) or "").strip()
    with st.expander(
        "🔑 Your OpenAI API key — "
        + ("✅ set for this session" if _saved else "⚠️ paste to run a live review"),
        # expanded depends ONLY on `running`, never on `_saved`. If it also
        # depended on `_saved`, typing a key (or clearing one) changes `_saved`
        # and snaps the panel open/closed on the very next rerun — that layout
        # shift eats whichever click triggered the rerun, so every action
        # looked like it needed two clicks.
        expanded=not running,
    ):
        st.caption(
            "Live Review calls OpenAI directly with **your** key. It's held "
            "**in memory for this browser session only** — never written to "
            "disk, never logged, never shared — and gone when you close the tab."
        )

        def _clear_key():
            # Read the widget's own session_state value directly (key=_wkey)
            # instead of mirroring it into a separate "byok_saved" variable via
            # on_change. Two callbacks racing on the same value (a save on
            # text-box blur, a clear on button click) can fire in either order —
            # if save lands after clear, it silently undoes the clear on the
            # very first click. Bumping the generation here is the only thing
            # this callback needs to do: next rerun reads a widget key that has
            # never existed, so the box renders blank.
            cur_gen = st.session_state.get("byok_gen", 0)
            st.session_state["byok_gen"] = cur_gen + 1
            st.session_state.pop(f"byok_input_{cur_gen}", None)

        st.text_input(
            "OpenAI key  (sk-…)",
            type="password",
            key=_wkey,
            disabled=running,
            help="Held in memory for this session only — never stored or logged.",
        )
        st.caption("Click **⏎ Enter** (or press Enter on your keyboard) to enable Run review.")
        col_use, col_clear = st.columns(2)
        with col_use:
            # A button click always triggers a Streamlit rerun on its own — this
            # button needs no on_click handler; clicking it just forces the page
            # to re-read the text box's current value, which is what unlocks Run.
            st.button("⏎ Enter", disabled=running, key="byok_use")
        with col_clear:
            st.button(
                "🧹 Clear key",
                disabled=running,
                key="byok_clear",
                on_click=_clear_key,
                help="Wipe the pasted key from this session's memory.",
            )

# Resolve which key to use (env-first, then this session's pasted key).
# resolve_key never writes os.environ — the key is forwarded explicitly to
# the agent (run_agent) and on to the OpenAI SDK.
_cur_gen = st.session_state.get("byok_gen", 0)
_session_key = (
    st.session_state.get(f"byok_input_{_cur_gen}", "") or ""
).strip() or None
api_key, key_source = resolve_key(_env_key, _session_key)

# Fast-fail on obvious junk (empty / too short / wrong prefix) BEFORE we spend
# an API call. Only relevant for a pasted key — never logs the key.
key_ok, key_msg = (True, "ok")
if key_source == "byok":
    key_ok, key_msg = validate_key_format(api_key)

# ── What to review ───────────────────────────────────────────────────────────
# GitHub-link only: a free-text local folder path would let a stranger on a
# deployed instance probe the server's own filesystem (path traversal), and
# it's meaningless anyway since the server can't see their laptop. A public
# repo URL has its .py files fetched via the GitHub API into a temp dir
# (fetcher.py), scanned, then deleted — local and deployed runs behave identically.
repo_url = st.text_input(
    "GitHub repo link",
    value="https://github.com/sriram98669704/code-review-sample",
    disabled=running,
    help="Paste a public GitHub repo URL (https://github.com/owner/repo). "
         "Only .py files are reviewed.",
)
parsed = parse_repo_url(repo_url)
target_ok = bool(parsed)

run = st.button(
    "▶  Run review", type="primary", key="run_btn",
    disabled=running or not api_key or not key_ok or not target_ok,
    help="Scan every Python function in the target repo. Makes live OpenAI "
         "calls and costs a few cents; progress streams live as it runs.",
)
if not running:
    if not api_key:
        st.caption("⚠️ Set your API key above to enable Run.")
    elif not key_ok:
        st.caption(f"⚠️ That key looks invalid ({key_msg}) — check it and re-paste.")
    elif not target_ok:
        st.caption("⚠️ That doesn't look like a public GitHub repo URL "
                    "(expected https://github.com/owner/repo).")

# ── PASS 1 — user clicked Run: stash the request, flip to running, rerun so
#    the controls above show disabled while PASS 2 does the work.
if run:
    st.session_state["review_pending"] = repo_url
    st.session_state["review_running"] = True
    st.rerun()

# ── PASS 2 — fetch the repo's .py files into a temp dir, execute inside
#    st.status, streaming the agent's own on_step messages live, then delete the
#    temp dir — exactly like llm-eval-lab's Live Test tab, but with a fetch step
#    around the review.
if running:
    pending = st.session_state.get("review_pending", "")
    steps = []

    _parsed = parse_repo_url(pending)
    _label = f"{_parsed[0]}/{_parsed[1]}" if _parsed else pending

    result, report, reads, error = None, None, [], None
    exploration = None
    try:
        with st.status("Running review…", expanded=True) as status:
            # One placeholder we re-render on every step, so the live progress
            # grows as the SAME nested tree the Run log shows afterwards —
            # instead of a flat stream of identical bullet lines.
            log_box = st.empty()

            def on_step(msg):
                steps.append(msg)
                with log_box.container():
                    render_agent_log(steps, repo_label=_label)

            with fetched_repo(pending) as repo_path:
                # The agent drives: it calls index_repo, judge, and read_function
                # itself, then a triage pass overrules the false positives; we get
                # back its verdict + the structured findings for the cards.
                out = run_agent(
                    f"Review the repository at '{repo_path}' and report its risks.",
                    api_key=api_key, on_step=on_step,
                )
                result, report, reads = out["review"], out["report"], out["reads"]
                exploration = out.get("exploration")
            _res = result or {"findings": [], "duplicates": [], "dropped": []}
            status.update(
                label=f"✅ Done — {len(_res['findings'])} findings, "
                      f"{len(_res.get('dropped', []))} overruled, "
                      f"{len(_res['duplicates'])} duplicates",
                state="complete",
            )
    except Exception as exc:
        error = redact(str(exc))

    st.session_state["review_result"] = {
        "result": result, "report": report, "reads": reads,
        "label": _label, "error": error, "steps": steps,
        "exploration": exploration,
    }
    st.session_state["review_running"] = False
    st.session_state.pop("review_pending", None)
    st.rerun()

# ── PASS 3 — render the last result (persists across reruns until the next run).
record = st.session_state.get("review_result") if not running else None
if record:
    if record["steps"]:
        with st.expander("🧾 Agent loop — every decision and the functions it walked",
                         expanded=True):
            st.caption(
                "The agent's path, step by step. It reviews like a **deliberately "
                "paranoid reviewer** — it flags anything that looks risky — and then a "
                "**separate triage pass overrules the false positives** by reading the "
                "actual code of the functions a flagged function calls. It has to be a "
                "*separate* pass: the reviewer that raised a finding tends to defend it, "
                "so a **fresh reviewer with no stake in the original verdict** is the one "
                "willing to drop it. What survives both passes is the real risk, not noise."
            )
            render_agent_log(record["steps"], done=True,
                             repo_label=record.get("label"),
                             dropped=(record.get("result") or {}).get("dropped", []),
                             exploration=record.get("exploration"))

    st.divider()
    if record["error"]:
        st.error(f"Error running review: `{record['error']}`")
    else:
        result = record["result"] or {"findings": [], "duplicates": [], "dropped": []}
        findings = result["findings"]
        duplicates = result["duplicates"]
        dropped = result.get("dropped", [])

        # Headline numbers, grouped so nobody adds them up. High/Medium/Low are a
        # BREAKDOWN of the risk-findings total; Overruled and Duplicates are SEPARATE
        # counts that are NOT part of that total.
        sev = {s: sum(1 for f in findings if f.get("severity") == s)
               for s in ("high", "medium", "low")}
        st.markdown("**Risk findings** — issues that survived triage")
        c = st.columns(4)
        c[0].metric("Total", len(findings),
                    help="Security + compliance (GRC) + guardrail issues left after the false "
                         "positives were overruled. The three severities beside it add up to this.")
        c[1].metric("🔴 High", sev["high"], help="Part of the total — the ones to fix first.")
        c[2].metric("🟠 Medium", sev["medium"], help="Part of the total.")
        c[3].metric("🟢 Low", sev["low"], help="Part of the total.")
        st.caption(f"🔴 {sev['high']} + 🟠 {sev['medium']} + 🟢 {sev['low']} = "
                   f"**{len(findings)}** — High/Medium/Low are a breakdown of the total, not extra.")
        st.markdown("**Counted separately** — not part of the risk findings above")
        c2 = st.columns(2)
        c2[0].metric("Overruled (false alarms dropped)", len(dropped),
                     help="Findings the first review raised, then triage proved safe by reading "
                          "the function the code calls.")
        c2[1].metric("Duplicate functions", len(duplicates),
                     help="Functions that are exact copies or near-identical to another.")
        render_severity_summary(findings)

        # Detailed finding cards next — the specifics behind the numbers above.
        st.divider()
        st.subheader(f"Findings ({len(findings)})")
        render_findings(findings)
        if dropped:
            st.subheader(f"Overruled — dropped false positives ({len(dropped)})")
            render_dropped(dropped, kept=findings)
        st.subheader(f"Duplicates ({len(duplicates)})")
        render_duplicates(duplicates)

        # The agent's narrative verdict LAST — a plain-English wrap-up of everything above.
        if record.get("report"):
            st.divider()
            st.subheader("🤖 Agent's risk verdict")
            st.caption(
                "The agent's own plain-English summary of everything above — including "
                "which findings it overruled and why. (The **Overruled** and **Duplicates** "
                "sections above are the structured view of the same run.)"
            )
            st.markdown(_strip_report_title(record["report"]))
        st.caption(HONESTY_LINE)
