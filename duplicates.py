"""Step 4 - the duplicate detector (now backed by Chroma).

Turn each function into a vector (its "meaning" as numbers), then stream through
the functions one at a time. For each one, search the functions we've ALREADY
seen for the most similar; if it's close enough, ask the LLM to confirm it's a
real copy before flagging. We search BEFORE we add the current function to the
pile, so a function never matches itself.

The pile is a Chroma collection - a real vector store. It holds every function's
vector and does the nearest-neighbour search for us (cosine distance). For a
handful of functions this is overkill, but it's the same code that scales to a
big repo: persistence and fast indexed search come for free.
"""

import json
import uuid
from pathlib import Path

import chromadb

from chunker import walk_files, chunk_file
from llm import chat, embed

THRESHOLD = 0.80          # the line we drew from the 4a table (twins were 0.86)

CONFIRM_SYSTEM = (
    "You compare two Python functions and decide if one is a redundant copy of "
    "the other - the same logic, possibly with renamed variables or functions. "
    "Two functions that merely share a topic are NOT duplicates. Answer in JSON."
)


def new_pile():
    """A fresh, ISOLATED vector store for one review run.

    Chroma's in-memory client is shared for the whole process, so two runs that
    reused the same collection name would see each other's functions - the second
    run would "find" leftovers from the first and flag them as duplicates. A
    unique name per run guarantees a clean pile every time; drop_pile() frees it
    when the run is done.

    `hnsw:space=cosine` makes Chroma rank by cosine distance, so the score we
    read back (1 - distance) is the same cosine similarity we used before."""
    store = chromadb.EphemeralClient()
    return store.get_or_create_collection(
        name=f"code_pile_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"}
    )


def drop_pile(pile):
    """Delete a run's pile so a long-running process (the agent loop, the UI)
    doesn't accumulate stale collections in memory."""
    chromadb.EphemeralClient().delete_collection(pile.name)


def confirm_duplicate(a, b, api_key=None):
    """Ask the LLM to confirm two close functions are really duplicates."""
    user = (
        f"Function A ({a['name']}):\n{a['code']}\n\n"
        f"Function B ({b['name']}):\n{b['code']}\n\n"
        'Are these duplicates (same logic, possibly renamed)? '
        'Return {"duplicate": true or false, "reason": "<one sentence>"}.'
    )
    resp = chat(
        messages=[{"role": "system", "content": CONFIRM_SYSTEM},
                  {"role": "user", "content": user}],
        api_key=api_key,
        response_format={"type": "json_object"},
        temperature=0,
        seed=0,                               # same dice every run -> repeatable
    )
    data = json.loads(resp.choices[0].message.content)
    return bool(data.get("duplicate", False)), data.get("reason", "")


def check_duplicate(chunk, pile, api_key=None, vec=None):
    """Check ONE chunk against the `pile` (a Chroma collection).

    Returns a finding dict if it's a confirmed duplicate, else None - then adds
    this chunk to the pile. We search BEFORE we add, so it can't match itself.
    This is the per-chunk step the streaming review loop calls.

    `vec` is the chunk's embedding if already computed - the index pass embeds each
    chunk once and shares it with the rule retrieval, so it passes the vector here
    to skip a redundant embed; omit it and we embed the chunk ourselves."""
    if vec is None:
        vec = embed(chunk["code"], api_key=api_key)

    best, best_score = None, 0.0              # most similar earlier function
    if pile.count() > 0:                      # nothing to compare against yet?
        res = pile.query(query_embeddings=[vec], n_results=1)
        best_score = 1.0 - res["distances"][0][0]          # cosine similarity
        best = {"name": res["metadatas"][0][0]["name"],
                "path": res["metadatas"][0][0]["path"],
                "code": res["documents"][0][0]}

    finding = None
    if best is not None and best_score >= THRESHOLD:       # above the line?
        is_dup, reason = confirm_duplicate(best, chunk, api_key=api_key)
        if is_dup:                                         # LLM agrees -> flag
            # Exact = byte-for-byte identical body (copy-paste); similar = same
            # logic with renamed vars / reordered lines. The UI surfaces both,
            # and which one it is, so a reviewer knows whether it's a literal
            # copy or a near-twin.
            exact = chunk["code"].strip() == best["code"].strip()
            finding = {
                "path": chunk["path"], "name": chunk["name"],
                "duplicate_of": best["name"], "duplicate_of_path": best["path"],
                "score": best_score, "reason": reason,
                "kind": "exact" if exact else "similar",
            }

    pile.add(                                 # remember it AFTER searching
        ids=[f"{chunk['path']}::{chunk['name']}::{chunk['start']}"],
        embeddings=[vec],
        metadatas=[{"name": chunk["name"], "path": chunk["path"]}],
        documents=[chunk["code"]],
    )
    return finding


def find_duplicates(chunks):
    """Standalone helper: stream chunks through check_duplicate, fresh pile."""
    pile, findings = new_pile(), []
    try:
        for chunk in chunks:
            f = check_duplicate(chunk, pile)
            if f:
                findings.append(f)
        return findings
    finally:
        drop_pile(pile)


if __name__ == "__main__":
    sample = Path(__file__).parent.parent / "code-review-sample"
    funcs = [c for path in walk_files(sample)
             for c in chunk_file(path) if c["kind"] == "code"]

    findings = find_duplicates(funcs)
    print(f"{len(findings)} duplicate(s):\n")
    for f in findings:
        rel = Path(f["path"]).relative_to(sample)
        print(f"  {rel} :: {f['name']} duplicates {f['duplicate_of']} "
              f"(score {f['score']:.2f})")
        print(f"       {f['reason']}")
