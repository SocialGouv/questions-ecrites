#!/usr/bin/env python3
"""Promote MIN15-discovered bureaus into the `bureaux` referential.

Rationale
---------
The `bureaux` referential holds the 36 DGCS/DSS bureaus seeded from the
org-chart documents those directions provided. The MIN15 extracts
surface ~127 more bureau keys — including the entire DGOS/DGS coverage —
that only exist as free text inside the `question_attributions_all`
view: invisible in the admin Organisation page, impossible to rename,
and rendered with regex-built labels ("SDRH1 — Chef de bureau").

This script closes the gap: every *legitimate* bureau reachable from
MIN15 data becomes a real `bureaux` row (statut='min15'), so the
referential is the single home of every bureau the application can
display, and the admin can curate labels in one place.

Matching & noise rules
----------------------
Keys come from the view itself (source='min15'), so normalisation is
guaranteed identical to what the attribution vote uses. Then:

1. COLLAPSE pool suffixes: 'SD1B/REDACTEURS' names the bureau SD1B, the
   'REDACTEURS' segment is a workflow pool, not an entity. Same for
   VALIDEURS and CHEF ('SDRH1/CHEF' = the chief's inbox of bureau
   SDRH1). The collapsed key keeps only the prefix.
2. SKIP sous-direction-level keys: 'SDPP/SOUS' ("Sous-direction") is an
   attribution to a sous-direction, not to a bureau — no entity to
   create.
3. LINK when the (possibly collapsed) key, or its post-'/' segment
   alone, matches an existing referential key (derived from the
   '[KEY]' prefix of `bureaux.nom`): the existing row gets its
   `min15_key` filled, nothing is created. This is how
   'SD1B/REDACTEURS' → '[SD1B] Relations avec…' and
   'SD1/MCGRM' → '[MCGRM] Mission de la coordination…' resolve.
4. CREATE otherwise, when the key has bureau granularity (contains a
   digit after collapsing — 'SDRH1' yes, bare 'SDAS' no) and is seen on
   at least --min-freq questions (default 3, filters one-off typos).
   The new row is named '[KEY] Thematic label' when the MIN15 label
   carries a thematic part ('Bureau SP3 - Prévention des addictions' →
   'Prévention des addictions'), or '[KEY]' alone otherwise — the whole
   point of promoting them is that an admin can now rename them.

Idempotent: `min15_key` is unique; existing keys are skipped on rerun.
Directions not present in the `directions` referential (DGE, DGCCRF…)
are out of the ministères-sociaux scope and only logged.

Usage
-----
    poetry run python scripts/sync_bureaux_from_min15.py
        [--dry-run]      print the plan, write nothing
        [--min-freq N]   minimum questions per key to create (default 3)
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass

from sqlalchemy import text as sqltext

from qe.db import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Workflow-pool suffixes that name a queue inside a bureau, not a
# distinct entity: collapse the key to its prefix.
COLLAPSE_SUFFIXES = {"REDACTEURS", "VALIDEURS", "CHEF"}

# Suffixes that mark a sous-direction-level (or purely administrative)
# attribution — no bureau entity behind them.
SKIP_SUFFIXES = {"SOUS", "CABINET", "COORDINATION", "SECRETARIAT", "SECRÉTARIAT"}

# Labels that describe a role or a queue, never a bureau's thematic
# name — rejected when picking the display label of a created row.
GENERIC_LABELS = {
    "chef de bureau", "chef de pôle", "sous-direction", "sous direction",
    "coordination", "rédacteurs", "redacteurs", "valideurs", "cabinet",
}

# "Bureau SP3 - Prévention des addictions" → "Prévention des addictions"
_THEMATIC_RE = re.compile(r"^\s*Bureau\s+\S+\s*-\s*(.+)$", re.IGNORECASE)


@dataclass
class Plan:
    linked: list[tuple[int, str, str]]  # (bureau_id, nom, min15_key)
    created: list[tuple[str, str, int, int]]  # (key, nom, direction_id, freq)
    skipped: list[tuple[str, str, int]]  # (key, reason, freq)


def _collapse(key: str) -> str | None:
    """Apply COLLAPSE / SKIP rules. None = not a bureau entity.

    Three collapse cases:
    - pool suffix ('SD1B/REDACTEURS') → the prefix is the bureau;
    - DGOS-style keys where the PREFIX already carries the bureau
      granularity ('SDAS1/ZONAGE', 'SDP3/SANTÉ'): the suffix is the
      first word of the bureau's thematic *name*, not a sub-entity —
      collapse to the prefix and let the label logic recover the
      thematic name;
    - digit-bearing suffixes ('SDRH1/1A', 'SDSP/SP3') stay as-is: the
      suffix is the actual bureau code.
    """
    if "/" not in key:
        return key
    prefix, suffix = key.rsplit("/", 1)
    if suffix in SKIP_SUFFIXES:
        return None
    if suffix in COLLAPSE_SUFFIXES:
        return prefix
    prefix_has_digit = any(ch.isdigit() for ch in prefix)
    suffix_has_digit = any(ch.isdigit() for ch in suffix)
    if prefix_has_digit and not suffix_has_digit:
        return prefix
    return key


def _thematic_label(label: str) -> str | None:
    """Extract the thematic half of a view label, if any.

    View labels look like 'SD SP — Bureau SP3 - Prévention des
    addictions' (sous_direction — bureau_full). Only the part after the
    em-dash can carry a thematic name.
    """
    part = label.split("—", 1)[1].strip() if "—" in label else label.strip()
    m = _THEMATIC_RE.match(part)
    if m:
        part = m.group(1).strip()
    if part.lower() in GENERIC_LABELS:
        return None
    # A part that is descriptive (longer than an acronym) is worth
    # keeping as-is; short leftovers ('2B', 'CM') are codes, not names.
    return part if len(part) > 8 else None


def _plan_entity(
    plan: Plan,
    direction_label: str,
    key: str,
    freq: int,
    labels: list[tuple[int, str]],
    by_ref_key: dict[str, object],
    taken_min15_keys: set[str],
    directions: dict[str, int],
    min_freq: int,
) -> None:
    """Decide LINK / CREATE / SKIP for one aggregated bureau entity."""
    if key in taken_min15_keys:
        plan.skipped.append((key, "déjà synchronisé (min15_key)", freq))
        return
    # LINK: full key, or its post-'/' segment, matches the referential.
    match = by_ref_key.get(key)
    if match is None and "/" in key:
        match = by_ref_key.get(key.rsplit("/", 1)[1])
    if match is not None:
        if match.min15_key is None:  # type: ignore[attr-defined]
            plan.linked.append((match.id, match.nom, key))  # type: ignore[attr-defined]
        else:
            plan.skipped.append((key, f"référentiel déjà lié ({match.nom})", freq))  # type: ignore[attr-defined]
        return
    # CREATE candidates.
    if direction_label not in directions:
        plan.skipped.append((key, f"direction hors référentiel ({direction_label})", freq))
        return
    bare = key.rsplit("/", 1)[1] if "/" in key else key
    if not any(ch.isdigit() for ch in bare):
        plan.skipped.append((key, "pas de granularité bureau (aucun chiffre)", freq))
        return
    if freq < min_freq:
        plan.skipped.append((key, f"trop rare (< {min_freq} questions)", freq))
        return
    # Several raw keys can fold into one entity ('SDAS1/CHEF' +
    # 'SDAS1/ZONAGE' → SDAS1); pick the most frequent label that yields
    # a real thematic name, so the pool variant never wins over the
    # descriptive one.
    thematic_candidates = [
        (label_freq, thematic)
        for label_freq, label in labels
        if (thematic := _thematic_label(label)) is not None
    ]
    thematic_best = max(thematic_candidates)[1] if thematic_candidates else None
    nom = f"[{key}] {thematic_best}" if thematic_best else f"[{key}]"
    plan.created.append((key, nom, directions[direction_label], freq))


def build_plan(min_freq: int) -> Plan:
    engine = get_engine()
    with engine.connect() as conn:
        # Existing referential: id + key derived from the '[KEY]' prefix,
        # exactly like the attribution view derives it.
        existing = conn.execute(
            sqltext("""
                SELECT b.id, b.nom, b.min15_key,
                       upper(replace((regexp_match(b.nom, '^\\s*\\[([^\\]]+)\\]'))[1], ' ', '')) AS ref_key
                FROM bureaux b
            """)
        ).fetchall()
        by_ref_key = {r.ref_key: r for r in existing if r.ref_key}
        taken_min15_keys = {r.min15_key for r in existing if r.min15_key}

        directions = {
            r.nom: r.id
            for r in conn.execute(sqltext("SELECT id, nom FROM directions")).fetchall()
        }

        # MIN15 universe, straight from the view so normalisation matches
        # the attribution vote byte-for-byte.
        rows = conn.execute(
            sqltext("""
                SELECT direction_label, bureau_key, bureau_label, COUNT(*) AS freq
                FROM question_attributions_all
                WHERE source = 'min15'
                GROUP BY 1, 2, 3
                ORDER BY freq DESC
            """)
        ).fetchall()

    plan = Plan(linked=[], created=[], skipped=[])
    # Aggregate per collapsed key: several raw keys/labels can fold into
    # one entity ('SDRH1/CHEF' + future 'SDRH1/...' variants).
    agg: dict[tuple[str, str], dict[str, object]] = {}
    for r in rows:
        collapsed = _collapse(r.bureau_key)
        if collapsed is None:
            plan.skipped.append((r.bureau_key, "sous-direction / administratif", r.freq))
            continue
        slot = agg.setdefault(
            (r.direction_label, collapsed),
            {"freq": 0, "labels": []},
        )
        slot["freq"] = int(slot["freq"]) + int(r.freq)  # type: ignore[arg-type]
        slot["labels"].append((int(r.freq), r.bureau_label))  # type: ignore[union-attr]

    for (direction_label, key), slot in sorted(
        agg.items(), key=lambda kv: -int(kv[1]["freq"])  # type: ignore[arg-type]
    ):
        _plan_entity(
            plan,
            direction_label,
            key,
            int(slot["freq"]),  # type: ignore[arg-type]
            slot["labels"],  # type: ignore[arg-type]
            by_ref_key,
            taken_min15_keys,
            directions,
            min_freq,
        )

    return plan


def apply_plan(plan: Plan) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        for bureau_id, _nom, key in plan.linked:
            conn.execute(
                sqltext(
                    "UPDATE bureaux SET min15_key = :key "
                    "WHERE id = :id AND min15_key IS NULL"
                ),
                {"key": key, "id": bureau_id},
            )
        for key, nom, direction_id, _freq in plan.created:
            conn.execute(
                sqltext(
                    "INSERT INTO bureaux (nom, direction_id, statut, min15_key) "
                    "VALUES (:nom, :direction_id, 'min15', :key) "
                    "ON CONFLICT (min15_key) DO NOTHING"
                ),
                {"nom": nom, "direction_id": direction_id, "key": key},
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-freq", type=int, default=3)
    args = parser.parse_args()

    plan = build_plan(args.min_freq)

    logger.info("── LIER au référentiel existant (%d) ──", len(plan.linked))
    for _id, nom, key in plan.linked:
        logger.info("  %-20s → %s", key, nom)
    logger.info("── CRÉER (%d) ──", len(plan.created))
    for key, nom, _did, freq in plan.created:
        logger.info("  %-20s %-70s (%d QE)", key, nom, freq)
    logger.info("── IGNORER (%d) ──", len(plan.skipped))
    for key, reason, freq in plan.skipped:
        logger.info("  %-20s %-45s (%d QE)", key, reason, freq)

    if args.dry_run:
        logger.info("Dry-run : aucune écriture.")
        return
    apply_plan(plan)
    logger.info(
        "Terminé : %d liés, %d créés.", len(plan.linked), len(plan.created)
    )


if __name__ == "__main__":
    main()
