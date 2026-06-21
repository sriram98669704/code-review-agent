"""github fetch: turn a public GitHub URL into a local folder to scan.

A GitHub link is just an address; to review the code we need the actual files
on disk. We do the simplest reliable thing: a shallow `git clone` into a
temporary directory, scan that, then delete it. Nothing persists - on a laptop
the temp dir lives in /tmp; on Streamlit Cloud it's in the server container's
/tmp, a different machine that never touches your laptop or the GitHub repo.

Two fetch paths, same contract (a context manager yielding a temp Path, deleted
on exit), so callers don't care which ran:

  * api_fetched_repo() - the lean default. Lists the repo's file tree in ONE API
    call, keeps only the .py files, and downloads each one (one call each) into a
    temp dir. It never pulls .git history or any non-Python file, so it moves far
    less data than a clone. A GITHUB_TOKEN env var raises the rate limit from 60
    to 5000 requests/hour; without one, small public repos still work.
  * cloned_repo() - a shallow `git clone` fallback. Simpler and rate-limit-free,
    but it downloads the WHOLE repo (history + every file type) to disk. Used
    automatically when the API path can't run (e.g. rate-limited with no token).

fetched_repo() prefers the API path and falls back to the clone. (A true
never-touches-disk, RAM-only variant was considered and rejected: the index,
duplicate pile and resolver all hold every function in memory for the whole run
anyway, so streaming saves nothing downstream while it would mean rewriting the
chunker/index/resolver - see README, "Fetching a repo".)

Security: only PUBLIC https://github.com/owner/repo URLs are accepted. We
validate the URL shape first and rebuild a canonical URL from the parsed
owner/repo. For the clone we pass it to git in list form (never a shell string)
with `--` before it so it can never be read as a git option; for the API we only
ever interpolate the parsed owner/repo into api.github.com paths. This matters
because on a deployed app a stranger controls that input box.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

# GitHub REST API base, request headers, and a generous ceiling on how many .py
# files we pull from one repo. The cap is far above any repo we actually review,
# so it never trips on our runs; it only stops a giant repo from firing thousands
# of blob requests at the rate limit.
_API = "https://api.github.com"
_GH_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
_MAX_PY_FILES = 1000

# https://github.com/<owner>/<repo>  (optional .git suffix, optional trailing /).
# GitHub owner/repo names allow letters, digits, '-', '_', '.'.
_REPO_RE = re.compile(
    r"^https://github\.com/"
    r"(?P<owner>[A-Za-z0-9][A-Za-z0-9_.-]*)/"
    r"(?P<repo>[A-Za-z0-9][A-Za-z0-9_.-]*?)"
    r"(?:\.git)?/?$"
)


def parse_repo_url(url):
    """Validate a public GitHub repo URL and return (owner, repo), or None.

    Rejects anything that isn't exactly https://github.com/owner/repo - no other
    host, no ssh/git/file scheme, no extra path segments. None means
    'do not clone this'.
    """
    if not url:
        return None
    m = _REPO_RE.match(url.strip())
    if not m:
        return None
    return m.group("owner"), m.group("repo")


@contextmanager
def cloned_repo(url, timeout=60):
    """Clone a public GitHub repo into a temp dir, yield its Path, delete after.

    Use as:
        with cloned_repo(url) as path:
            run_agent(f"Review the repository at '{path}'...", api_key=...)

    Raises ValueError for a URL we won't accept, RuntimeError if the clone
    itself fails (private/missing repo, network, timeout). The temp dir is
    always removed on exit, even on error.
    """
    parsed = parse_repo_url(url)
    if not parsed:
        raise ValueError(
            "not a public GitHub repo URL (expected https://github.com/owner/repo)"
        )
    owner, repo = parsed
    clean_url = f"https://github.com/{owner}/{repo}.git"  # canonical, from parsed parts

    tmpdir = tempfile.mkdtemp(prefix="review_")
    try:
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--", clean_url, tmpdir],
                check=True, capture_output=True, text=True, timeout=timeout,
            )
        except FileNotFoundError:
            raise RuntimeError("git is not installed on this machine")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"git clone timed out after {timeout}s")
        except subprocess.CalledProcessError as e:
            last = (e.stderr or "").strip().splitlines()
            raise RuntimeError(f"git clone failed: {last[-1] if last else 'unknown error'}")
        yield Path(tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _gh_json(url, token, timeout):
    """GET a GitHub API URL and parse the JSON body. Raises RuntimeError with a clear,
    user-facing message on a rate limit (suggesting a token), a missing/private repo,
    or network trouble - never a raw stack trace into the deployed UI."""
    req = urllib.request.Request(url, headers=dict(_GH_HEADERS))
    if token:                                       # bearer auth -> 5000 req/hr instead of 60
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 403 and e.headers.get("X-RateLimit-Remaining") == "0":
            raise RuntimeError(
                "GitHub API rate limit reached - set a GITHUB_TOKEN environment "
                "variable (a free personal access token) to raise it to 5000 "
                "requests/hour")
        if e.code == 404:
            raise RuntimeError("repo not found (private, or the URL is mistyped?)")
        raise RuntimeError(f"GitHub API error {e.code}")
    except (socket.timeout, TimeoutError):
        raise RuntimeError(f"GitHub API request timed out after {timeout}s")
    except urllib.error.URLError as e:
        raise RuntimeError(f"could not reach GitHub: {e.reason}")


@contextmanager
def api_fetched_repo(url, token=None, timeout=30):
    """Fetch ONLY the .py files of a public GitHub repo via the REST API, into a temp
    dir; yield its Path; delete it after. Same contract as cloned_repo(), but it never
    downloads .git history or non-Python files: it reads the file tree (one request),
    keeps the .py blobs, and fetches each one (one request each).

    `token` (or a GITHUB_TOKEN env var) raises the rate limit from 60 to 5000
    requests/hour; without one, small public repos still work. Raises ValueError for a
    URL we won't accept, RuntimeError if an API call fails. The temp dir is always
    removed on exit, even on error."""
    parsed = parse_repo_url(url)
    if not parsed:
        raise ValueError(
            "not a public GitHub repo URL (expected https://github.com/owner/repo)"
        )
    owner, repo = parsed                            # only the parsed parts touch the API paths
    token = token or os.environ.get("GITHUB_TOKEN")

    meta = _gh_json(f"{_API}/repos/{owner}/{repo}", token, timeout)
    branch = meta.get("default_branch", "main")     # no branch-guessing: ask the repo
    tree = _gh_json(
        f"{_API}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1", token, timeout)
    py = [e for e in tree.get("tree", [])           # one tree call lists every file...
          if e.get("type") == "blob" and e.get("path", "").endswith(".py")]
    if not py:                                      # ...we keep only the Python ones
        raise RuntimeError("repo has no Python files to review")
    py = py[:_MAX_PY_FILES]                          # generous cap; protects the rate limit

    tmpdir = tempfile.mkdtemp(prefix="review_")
    try:
        for e in py:                                # one blob request per .py file
            blob = _gh_json(
                f"{_API}/repos/{owner}/{repo}/git/blobs/{e['sha']}", token, timeout)
            content = base64.b64decode(blob.get("content", ""))  # blobs arrive base64
            dest = Path(tmpdir) / e["path"]         # recreate the repo-relative path so the
            dest.parent.mkdir(parents=True, exist_ok=True)       # scanner reads it like any
            dest.write_bytes(content)               # folder - no downstream change needed
        yield Path(tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@contextmanager
def fetched_repo(url, timeout=30):
    """Prefer the lean .py-only API fetch; fall back to a shallow clone if the API path
    can't run (e.g. rate-limited with no token, or a transient API error). A bad URL
    (ValueError) is NOT a fallback case - both paths reject it - so it propagates.

    We enter the API context manager explicitly so the fallback only fires on a SETUP
    failure (everything before its yield); an error raised by the body still propagates
    normally and never silently re-runs the review on a clone."""
    cm = api_fetched_repo(url, timeout=timeout)
    try:
        path = cm.__enter__()                       # runs the API fetch
    except RuntimeError:                            # API couldn't produce files -> clone instead
        with cloned_repo(url, timeout=timeout) as path:
            yield path
        return
    try:
        yield path
    finally:
        cm.__exit__(None, None, None)               # always delete the API temp dir
