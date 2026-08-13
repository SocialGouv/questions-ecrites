"""Guard: the Alembic history must stay a single linear chain.

`alembic upgrade head` (singular) is the documented and deployed command
(README, CLAUDE.md, scripts/load_pgvector.py). It aborts with "Multiple
head revisions are present" as soon as two migrations declare the same
``down_revision`` and both land on main.

Several feature branches add migrations in parallel, so this is a real
failure mode: each branch is fine on its own, and the breakage only
appears once the second one merges. GitHub runs PR checks against the
prospective merge commit, so asserting it here turns a post-merge
production incident into a red check on the second PR — at which point
the fix is a one-line ``down_revision`` re-chain (or ``alembic merge``).
"""

from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]


def _script_directory() -> ScriptDirectory:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    return ScriptDirectory.from_config(config)


def test_single_alembic_head() -> None:
    heads = _script_directory().get_heads()
    assert len(heads) == 1, (
        "Alembic history has forked into several heads: "
        f"{sorted(heads)}. `alembic upgrade head` cannot run. Re-chain the "
        "newest migration's down_revision onto the other head."
    )


def test_every_revision_resolves() -> None:
    """No migration may point at a down_revision that does not exist.

    Catches the mirror-image mistake of the fork: re-chaining a branch
    onto a revision that only exists in another, unmerged branch.
    """
    script = _script_directory()
    known = {revision.revision for revision in script.walk_revisions()}
    dangling = {
        revision.revision: revision.down_revision
        for revision in script.walk_revisions()
        if revision.down_revision is not None
        and not set(
            revision.down_revision
            if isinstance(revision.down_revision, tuple)
            else (revision.down_revision,)
        )
        <= known
    }
    assert not dangling, f"Migrations pointing at unknown parents: {dangling}"
