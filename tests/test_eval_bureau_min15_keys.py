"""The MIN15 bureau key rule, in both places it is written.

`canonical_from_extract` (eval script) and `BUREAU_KEY_SQL` (migration
5b1c9e2d7a4f, the rule production actually votes on) are hand-maintained
duplicates. `KEY_CASES` below is the single expected mapping and both tests
assert against it — the second one by running the migration's SQL in
PostgreSQL — so editing one side alone fails here instead of silently making
the eval predict a ranking production no longer produces.
"""

import importlib.util
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from scripts.eval_bureau_with_min15 import canonical_from_extract

MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "alembic"
    / "versions"
    / "5b1c9e2d7a4f_min15_bureau_keys_at_bureau_level.py"
)

# (sous_direction, bureau, expected key) — None means "no bureau, drop the row".
KEY_CASES = [
    ("SD2", "Bureau 2B", "SD2/2B"),
    ("SD SP", "Bureau SP3", "SDSP/SP3"),
    ("SD1 B", "REDACTEURS", "SD1B"),
    ("SDRH1", "Chef de bureau", "SDRH1"),
    ("SDAS1", "Zonage, régulation territoriale", "SDAS1"),
    ("SD1", "MCGRM", "MCGRM"),
    ("DACI", "REDACTEURS", "DACI"),
    ("DACI", "Chargée de mission", "DACI"),
    ("SDAS", "Sous-Direction", None),
    ("SD3", "CAB S/DIR", None),
    ("CAB", "Cabinet", None),
    ("Centralisateur", "MDI", "CENTRALISATEUR/MDI"),
    # Bureau named in free text under a plain sous-direction (DGS, DGE): keeps a
    # key, only role and level labels drop the row.
    ("SD SP", "Pharmacie", "SDSP/PHARMACIE"),
    ("SDP", "Médecine de ville", "SDP/MÉDECINE"),
    ("SD1", "Chef de bureau", None),
    ("SD2", None, None),
]


@pytest.mark.parametrize(("sous_direction", "bureau", "expected"), KEY_CASES)
def test_min15_key_is_the_bureau_not_the_role(sous_direction, bureau, expected):
    assert canonical_from_extract(sous_direction, bureau) == expected


def _migration_bureau_key_sql() -> str:
    spec = importlib.util.spec_from_file_location(
        "_min15_bureau_key_migration", MIGRATION_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BUREAU_KEY_SQL


@pytest.mark.integration
def test_view_sql_key_rule_matches_its_python_mirror():
    """Same inputs, same keys — run against PostgreSQL, skipped without one."""
    from qe import db

    try:
        connection = db.get_engine().connect()
    except OperationalError as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no PostgreSQL available: {exc}")

    # BUREAU_KEY_SQL reads e.sous_direction / e.bureau: give it a one-row `e`.
    query = text(
        "SELECT " + _migration_bureau_key_sql() + " AS bureau_key "
        "FROM (VALUES (CAST(:sous_direction AS text), CAST(:bureau AS text)))"
        " AS e(sous_direction, bureau)"
    )
    with connection:
        mismatches = [
            (sous_direction, bureau, expected, observed)
            for sous_direction, bureau, expected in KEY_CASES
            if (
                observed := connection.execute(
                    query, {"sous_direction": sous_direction, "bureau": bureau}
                ).scalar_one()
            )
            != expected
        ]
    assert not mismatches, "the view's SQL rule drifted from KEY_CASES: " + repr(
        mismatches
    )
