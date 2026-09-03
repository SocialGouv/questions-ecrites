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
from typing import NamedTuple

from sqlalchemy import text as sqltext

from qe.attributions import refresh_attributions_all_view
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

# A candidate thematic label shorter than this is a bare code ('2B',
# 'CM', 'SP3') left over after stripping the 'Bureau X -' prefix, not a
# human-readable name — the longest real thematic name observed in the
# corpus below this bound is 'Médicament' (10 chars), the shortest
# code-noise above chance is 'MCGRM' (5).
MIN_THEMATIC_LEN = 9


class RefBureau(NamedTuple):
    """One existing `bureaux` row, keyed for the LINK lookup."""

    id: int
    nom: str
    min15_key: str | None


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
    # keeping as-is; shorter leftovers are codes, not names.
    return part if len(part) >= MIN_THEMATIC_LEN else None


def _find_referential_match(
    key: str,
    raw_keys: list[str],
    by_ref_key: dict[str, RefBureau],
) -> RefBureau | None:
    """LINK lookup: the collapsed key, its post-'/' segment, or the
    post-'/' segment of any RAW key folded into this entity.

    The raw-key pass matters when the collapse stole a suffix that was
    itself a referential bureau: 'SD1/MCGRM' collapses to 'SD1' (digit
    prefix rule) but must link to '[MCGRM]', not create a phantom
    '[SD1]'.
    """
    candidates = [key]
    if "/" in key:
        candidates.append(key.rsplit("/", 1)[1])
    candidates.extend(raw.rsplit("/", 1)[1] for raw in raw_keys if "/" in raw)
    for candidate in candidates:
        found = by_ref_key.get(candidate)
        if found is not None:
            return found
    return None


def plan_linked_ids(plan: Plan) -> set[int]:
    """Referential rows already claimed by a LINK earlier in this plan.

    The same collapsed key can surface under two direction labels
    (MIN15 direction text is free-form); without this guard both
    entities would link — or create — the same row twice.
    """
    return {bureau_id for bureau_id, _nom, _key in plan.linked}


def plan_created_keys(plan: Plan) -> set[str]:
    """Keys already scheduled for creation earlier in this plan."""
    return {key for key, _nom, _did, _freq in plan.created}


def _pick_nom(
    key: str,
    labels: list[tuple[int, str]],
    direction_label: str,
    raw_labels: dict[tuple[str, str], list[tuple[int, str]]],
) -> str:
    """Display name for a created bureau: '[KEY] Thematic label'.

    Several raw keys can fold into one entity ('SDAS1/CHEF' +
    'SDAS1/ZONAGE' → SDAS1); the most frequent label that yields a real
    thematic name wins, so the pool variant never beats the descriptive
    one. Collapsed-prefix entities whose view labels were all generic
    ('Chef de bureau') fall back to the raw-step harvest, where the
    drafting postes carry the real name.
    """
    thematic_candidates = [
        (label_freq, thematic)
        for label_freq, label in labels
        if (thematic := _thematic_label(label)) is not None
    ]
    best = max(thematic_candidates)[1] if thematic_candidates else None
    if best is None and "/" not in key:
        fallback = raw_labels.get((direction_label, key), [])
        if fallback:
            best = max(fallback)[1]
    return f"[{key}] {best}" if best else f"[{key}]"


def _plan_entity(
    plan: Plan,
    direction_label: str,
    key: str,
    freq: int,
    labels: list[tuple[int, str]],
    raw_keys: list[str],
    by_ref_key: dict[str, RefBureau],
    taken_min15_keys: set[str],
    directions: dict[str, int],
    min_freq: int,
    raw_labels: dict[tuple[str, str], list[tuple[int, str]]],
) -> None:
    """Decide LINK / CREATE / SKIP for one aggregated bureau entity."""
    if key in taken_min15_keys:
        plan.skipped.append((key, "déjà synchronisé (min15_key)", freq))
        return
    match = _find_referential_match(key, raw_keys, by_ref_key)
    if match is not None:
        if match.min15_key is None and match.id not in plan_linked_ids(plan):
            plan.linked.append((match.id, match.nom, key))
        else:
            plan.skipped.append((key, f"référentiel déjà lié ({match.nom})", freq))
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
    if key in plan_created_keys(plan):
        plan.skipped.append((key, "déjà créé sous une autre direction", freq))
        return
    # A slashed key whose PREFIX is itself already a known bureau
    # ('SDRH1/1A' when SDRH1 exists) names a cell inside that bureau,
    # not a distinct bureau — creating both would duplicate the entity
    # in the referential. Prefix-first ordering is guaranteed by the
    # freq sort: the parent bureau always aggregates more questions
    # than any one of its cells.
    if "/" in key:
        prefix = key.rsplit("/", 1)[0]
        if (
            prefix in taken_min15_keys
            or prefix in plan_created_keys(plan)
            or any(linked_key == prefix for _i, _n, linked_key in plan.linked)
        ):
            plan.skipped.append((key, f"sous-entité du bureau {prefix}", freq))
            return
    nom = _pick_nom(key, labels, direction_label, raw_labels)
    plan.created.append((key, nom, directions[direction_label], freq))


def _harvest_raw_labels(conn) -> dict[tuple[str, str], list[tuple[int, str]]]:
    """Thematic label candidates straight from the RAW workflow steps.

    The view (and `question_bureau_extract` before it) keeps only each
    question's LATEST bureau step — which in the DGOS circuit is almost
    always the chief's validation ('SDP4 - Chef de bureau'), shadowing
    the drafting poste that carries the actual name ('SDP4 - Relations
    usagers et expérience patient'). Harvesting every step of
    `reponses_extract_etapes` recovers those names.

    Returns {(direction, SD_KEY): [(freq, label), …]} where SD_KEY is
    the sous-direction segment normalised exactly like the view does
    (upper, spaces stripped) — the match target for collapsed entities.
    For multi-cell bureaus the most frequent cell name wins: a slightly
    off title beats a bare code, and admins can rename (product call).
    """
    rows = conn.execute(
        sqltext(
            "SELECT poste_etape, COUNT(*) AS freq "
            "FROM reponses_extract_etapes "
            "WHERE poste_etape IS NOT NULL "
            "GROUP BY poste_etape"
        )
    ).fetchall()
    split_re = re.compile(r"\s+-\s+")
    harvest: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for r in rows:
        segs = [x.strip() for x in split_re.split(r.poste_etape.strip()) if x.strip()]
        if len(segs) < 3:
            continue
        direction = segs[0].upper()
        sd_key = segs[1].upper().replace(" ", "")
        label = _thematic_label(" - ".join(segs[2:]))
        if label is None:
            continue
        harvest.setdefault((direction, sd_key), []).append((int(r.freq), label))
    return harvest


def build_plan(min_freq: int) -> Plan:
    # This function reads question_attributions_all below — refresh first so
    # the plan isn't built against a stale snapshot.
    refresh_attributions_all_view()
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

        raw_labels = _harvest_raw_labels(conn)

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
            {"freq": 0, "labels": [], "raw_keys": []},
        )
        slot["freq"] = int(slot["freq"]) + int(r.freq)  # type: ignore[arg-type]
        slot["labels"].append((int(r.freq), r.bureau_label))  # type: ignore[union-attr]
        slot["raw_keys"].append(r.bureau_key)  # type: ignore[union-attr]

    for (direction_label, key), slot in sorted(
        agg.items(), key=lambda kv: -int(kv[1]["freq"])  # type: ignore[arg-type]
    ):
        _plan_entity(
            plan,
            direction_label,
            key,
            int(slot["freq"]),  # type: ignore[arg-type]
            slot["labels"],  # type: ignore[arg-type]
            slot["raw_keys"],  # type: ignore[arg-type]
            by_ref_key,
            taken_min15_keys,
            directions,
            min_freq,
            raw_labels,
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
