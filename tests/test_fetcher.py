"""Offline tests for fetcher.parse_repo_url - the URL-validation gate. Pure regex,
no network, no cost. Run free, any time:

    /usr/bin/python3 tests/test_fetcher.py

This is the security boundary: on a deployed app a stranger controls the repo URL
box, so we accept ONLY https://github.com/owner/repo (plus an optional GitHub
/tree/<branch>/<subdir> suffix) and rebuild a canonical URL from the parsed parts.
These tests pin what is accepted and - more importantly - what is rejected (other
hosts, ssh/git schemes, non-tree path segments, '..' traversal, junk).

The two fetch paths themselves (api_fetched_repo / cloned_repo / _gh_json) hit the
live GitHub API or git, so they aren't unit-testable offline - they're exercised
by a real run, not here.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fetcher import parse_repo_url, RepoTarget

CASES = []
def case(name):
    def wrap(fn):
        CASES.append((name, fn))
        return fn
    return wrap


# ---- accepted: https://github.com/owner/repo, plus an optional /tree/<branch>/<subdir> ----

@case("plain owner/repo URL parses to a RepoTarget with no branch/subdir")
def _():
    assert parse_repo_url("https://github.com/octocat/Hello-World") == \
        RepoTarget("octocat", "Hello-World", None, None)

@case("a trailing .git suffix is stripped")
def _():
    assert parse_repo_url("https://github.com/octocat/Hello-World.git") == \
        RepoTarget("octocat", "Hello-World", None, None)

@case("a trailing slash is allowed")
def _():
    assert parse_repo_url("https://github.com/octocat/Hello-World/") == \
        RepoTarget("octocat", "Hello-World", None, None)

@case("surrounding whitespace is trimmed before matching")
def _():
    assert parse_repo_url("  https://github.com/octocat/Hello-World  ") == \
        RepoTarget("octocat", "Hello-World", None, None)

@case("dots, dashes and underscores in names are allowed")
def _():
    assert parse_repo_url("https://github.com/my-org/some_repo.tool") == \
        RepoTarget("my-org", "some_repo.tool", None, None)

@case("a /tree/<branch> suffix scopes to that branch, no subdir")
def _():
    assert parse_repo_url("https://github.com/octocat/Hello-World/tree/main") == \
        RepoTarget("octocat", "Hello-World", "main", None)

@case("a /tree/<branch>/<subdir> suffix scopes to one folder")
def _():
    assert parse_repo_url("https://github.com/fportantier/vulpy/tree/master/bad") == \
        RepoTarget("fportantier", "vulpy", "master", "bad")

@case("a nested subdir under /tree/<branch> is kept whole")
def _():
    assert parse_repo_url("https://github.com/o/r/tree/main/src/pkg") == \
        RepoTarget("o", "r", "main", "src/pkg")


# ---- rejected: anything that isn't exactly that shape -> None (do not fetch) ----

@case("None / empty -> None")
def _():
    assert parse_repo_url(None) is None
    assert parse_repo_url("") is None

@case("http (not https) is rejected")
def _():
    assert parse_repo_url("http://github.com/octocat/Hello-World") is None

@case("a non-github host is rejected")
def _():
    assert parse_repo_url("https://gitlab.com/octocat/Hello-World") is None
    assert parse_repo_url("https://github.evil.com/octocat/Hello-World") is None

@case("an ssh / git scheme is rejected")
def _():
    assert parse_repo_url("git@github.com:octocat/Hello-World.git") is None
    assert parse_repo_url("ssh://git@github.com/octocat/Hello-World") is None

@case("a bare owner with no repo is rejected")
def _():
    assert parse_repo_url("https://github.com/octocat") is None

@case("non-tree path segments (blob/issues) are rejected")
def _():
    assert parse_repo_url("https://github.com/octocat/Hello-World/blob/main/x.py") is None
    assert parse_repo_url("https://github.com/octocat/Hello-World/issues/1") is None

@case("a query string or fragment is rejected")
def _():
    assert parse_repo_url("https://github.com/octocat/Hello-World?tab=readme") is None
    assert parse_repo_url("https://github.com/octocat/Hello-World#readme") is None

@case("a name that doesn't start alphanumerically is rejected")
def _():
    assert parse_repo_url("https://github.com/-octocat/Hello-World") is None

@case("a path-traversal-looking host/path can't slip through")
def _():
    assert parse_repo_url("https://github.com/../../etc/passwd") is None

@case("a '..' traversal segment inside the subdir is rejected")
def _():
    assert parse_repo_url("https://github.com/o/r/tree/main/../etc") is None
    assert parse_repo_url("https://github.com/o/r/tree/main/a/../b") is None


def main():
    passed = 0
    for name, fn in CASES:
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL  {name}\n      {e}")
        except Exception as e:                          # noqa: BLE001
            print(f"ERROR {name}\n      {type(e).__name__}: {e}")
        else:
            print(f"pass  {name}")
            passed += 1
    print(f"\n{passed}/{len(CASES)} passed")
    sys.exit(0 if passed == len(CASES) else 1)


if __name__ == "__main__":
    main()
