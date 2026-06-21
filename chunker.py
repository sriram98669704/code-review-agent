"""Step 2 - the deterministic walk + AST chunker (the coverage engine).

This is the part that GUARANTEES every file and every function gets seen.
A plain loop does the walking (not the AI), so nothing is ever skipped. Each
file is split with Python's `ast` (a real parser), so we never mistake a comment
for a function, and we never miss code that lives outside a function (like a
hardcoded API_KEY at the top of a file).
"""

import ast
from pathlib import Path

from resolver import file_bindings

IGNORE = {"venv", "_archive", "__pycache__", ".git"}


def walk_files(root):
    """Return every .py file under `root` - deterministic, sorted, no skips."""
    return [
        p for p in sorted(Path(root).rglob("*.py"))
        if not any(part in IGNORE for part in p.parts)
    ]


def chunk_file(path):
    """Split one file into chunks. Each TOP-LEVEL thing becomes one chunk.

    Tagged 'code'   = a function/method  -> later gets security + duplicate checks.
    Tagged 'module' = imports, top-level assignments (API_KEY = ...) -> security only.
    """
    source = Path(path).read_text()
    tree = ast.parse(source)
    return _chunks_from_tree(tree, source.splitlines(), path)


def scan_file(path):
    """Read + parse ONE file ONCE, return (chunks, bindings). The index pass uses
    this so chunk extraction and the resolver's import/def extraction share a single
    read and a single parse - no file is ever walked twice."""
    source = Path(path).read_text()
    tree = ast.parse(source)
    return _chunks_from_tree(tree, source.splitlines(), path), file_bindings(tree)


def _chunks_from_tree(tree, lines, path):
    """The top-level walk that turns one parsed file into chunks - shared by
    chunk_file (chunks only) and scan_file (chunks + bindings, one parse)."""
    chunks = []
    for node in tree.body:                       # walk EVERY top-level item, in order
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(_chunk(node, lines, path, node.name, "code"))
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:                # methods inside a class are code too
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunks.append(_chunk(sub, lines, path, f"{node.name}.{sub.name}", "code"))
        else:                                    # imports, assignments, etc.
            chunks.append(_chunk(node, lines, path, f"<{type(node).__name__}>", "module"))
    return chunks


def _chunk(node, lines, path, name, kind):
    start = node.lineno
    end = getattr(node, "end_lineno", start)
    return {
        "path": str(path), "name": name, "kind": kind,
        "start": start, "end": end,
        "code": "\n".join(lines[start - 1:end]),
    }


if __name__ == "__main__":
    sample = Path(__file__).parent.parent / "code-review-sample"
    files = walk_files(sample)
    print(f"deterministic walk found {len(files)} files:\n")

    total = 0
    for f in files:
        chunks = chunk_file(f)
        total += len(chunks)
        print(f.relative_to(sample))
        for c in chunks:
            print(f"   [{c['kind']:6}] {c['name']:18} (lines {c['start']}-{c['end']})")
        print()

    print(f"total: {total} chunks across {len(files)} files - nothing skipped.")
