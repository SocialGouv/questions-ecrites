"""Analyseur de questions écrites JO — extraction du contexte et de la question réelle.

Une QE typique a 3 parties :
    [Ouverture, 100-300 c]  « M./Mme X attire l'attention de Y sur [CONTEXTE]. »
    [Contexte, 500-2500 c]  statistiques, ancrages locaux, citations, background
    [Question, 200-500 c]   « Il/elle lui demande / souhaite savoir… »

Ce module est PUR — pas de DB, pas de fichier, pas de I/O. Il expose une
fonction `parse(text)` qui renvoie un `ParsedQuestion`.

Les patterns sont des constantes du module : facile à ajouter/modifier une
formulation sans toucher au reste.

Cas particulier : les « rappels » (relances administratives : `rappelle à …
les termes de sa question n° XXX`) sont détectés à part — ce ne sont pas
de nouvelles questions et ils apportent peu de signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
# NB: [’''] tolère les apostrophes courbes ou droites des exports JO.

# Rappel administratif (n'est pas une nouvelle question).
RE_RAPPEL = re.compile(
    r"\brappelle\s+à\s+.+?\bles\s+termes\s+de\s+sa\s+question\b",
    re.IGNORECASE | re.DOTALL,
)

# Ouvertures — chaque pattern DOIT exposer un groupe nommé `contexte`.
# On les cherche dans les 600 premiers caractères.
OPENER_HEAD_CHARS = 600
# La partie contexte se termine sur : point suivi d'un caractère blanc puis
# majuscule (nouvelle phrase), ou fin de texte. `\s+` sous DOTALL absorbe
# aussi les `\r\n` qu'on trouve fréquemment dans les exports JO.
_CTX_END = r"(?=\.\s+[A-ZÀ-Ý]|\.\s*$|\.\Z)"
_CONNECTORS = r"(?:sur|concernant|au\s+sujet\s+de|à\s+propos\s+de|quant\s+à|relative(?:ment)?\s+à|relatifs?\s+à)"

OPENERS: list[tuple[str, re.Pattern[str]]] = [
    # Variantes autour de "attention"
    (
        "attire/appelle l'attention",
        re.compile(
            rf"(?:attire|appelle)\s+l['’]attention\s+d[e'’].+?\b{_CONNECTORS}\s+"
            rf"(?P<contexte>.+?){_CTX_END}",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "souhaite attirer/appeler l'attention",
        re.compile(
            rf"souhaite\s+(?:attirer|appeler)\s+l['’]attention\s+d[e'’].+?\b{_CONNECTORS}\s+"
            rf"(?P<contexte>.+?){_CTX_END}",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    # Verbes seuls d'accroche
    (
        "interroge",
        re.compile(
            rf"\binterroge\s+.+?\b{_CONNECTORS}\s+"
            rf"(?P<contexte>.+?){_CTX_END}",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "alerte",
        re.compile(
            rf"\balerte\s+.+?\b{_CONNECTORS}\s+"
            rf"(?P<contexte>.+?){_CTX_END}",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "questionne",
        re.compile(
            rf"\bquestionne\s+.+?\b{_CONNECTORS}\s+"
            rf"(?P<contexte>.+?){_CTX_END}",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "expose à … que",
        re.compile(
            rf"\bexpose\s+à\s+.+?\bque\s+"
            rf"(?P<contexte>.+?){_CTX_END}",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "demande à … de",
        re.compile(
            rf"\bdemande\s+à\s+.+?\bde\s+"
            rf"(?P<contexte>.+?){_CTX_END}",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "aux fins de connaître",
        re.compile(
            rf"aux\s+fins\s+de\s+connaître\s+"
            rf"(?P<contexte>.+?){_CTX_END}",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
]

# Clôtures — verbes qui introduisent la vraie question.
# On les cherche dans les 900 derniers caractères et on prend l'occurrence
# la plus PRÉCOCE (pour attraper le début de la vraie question, pas ses
# éventuelles répétitions).
CLOSER_TAIL_CHARS = 900
CLOSERS: list[tuple[str, re.Pattern[str]]] = [
    # Verbes de demande — pronom + verbe conjugué
    ("il/elle demande",
     re.compile(r"\b(?:il|elle)\s+(?:lui\s+|se\s+)?demande(?:ra|rait)?\b", re.IGNORECASE)),
    ("il/elle souhaite",
     re.compile(r"\b(?:il|elle)\s+souhait(?:e|erait|ait|erais)\b", re.IGNORECASE)),
    ("il/elle voudrait/aimerait",
     re.compile(r"\b(?:il|elle)\s+(?:voudrait|aimerait|prie|sollicite)\b", re.IGNORECASE)),
    ("il/elle interroge",
     re.compile(r"\b(?:il|elle)\s+(?:l['’])?interroge\b", re.IGNORECASE)),
    ("il/elle remercie",
     re.compile(r"\b(?:il|elle)\s+(?:la|le)?\s*remercie\b", re.IGNORECASE)),
    ("il/elle propose/attend",
     re.compile(r"\b(?:il|elle)\s+(?:lui\s+)?(?:propose|attend)\b", re.IGNORECASE)),
    # Formes inversées : « souhaite-t-elle », « souhaiterait-il », « pourrait-il »
    ("inversion souhaite/pourrait/aimerait -t-il/elle",
     re.compile(
         r"\b(?:souhait(?:e|erait|ait)|voudrait|aimerait|pourrait|entend|envisage|compte)"
         r"[-\s]t[-\s](?:il|elle)\b",
         re.IGNORECASE,
     )),
    # Ouvertures de sous-question tardive
    ("aussi/enfin, …",
     re.compile(r"\b(?:aussi|enfin)[,\s]+(?:qu[ei]|comment|dans|il|elle|le\s+ministre)\b", re.IGNORECASE)),
    # Interrogatives fréquentes en fin
    ("que compte/comptent",
     re.compile(r"\bqu[ei]\s+compt(?:e|ent)[-\s]t[-\s](?:il|elle|ils|elles)\b", re.IGNORECASE)),
]


# ---------------------------------------------------------------------------
# Résultat
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParsedQuestion:
    """Résultat de l'analyse d'une QE.

    Les champs sont *bruts* : c'est à la couche appelante de décider comment
    les stocker ou les combiner (embedding, affichage, filtres).
    """
    est_rappel: bool
    """True si le texte est une relance administrative (référence explicite
    à une question antérieure)."""

    contexte_extrait: str | None
    """Le contexte de la question, extrait de l'accroche d'ouverture, ou None
    si aucun pattern n'a matché."""

    question_extraite: str | None
    """La question réelle finale, extraite du dernier paragraphe, ou None
    si aucun pattern de clôture n'a matché."""

    opener_label: str | None
    """Étiquette du pattern d'ouverture qui a matché (utile pour stats)."""

    closer_label: str | None
    """Étiquette du pattern de clôture qui a matché (utile pour stats)."""


# ---------------------------------------------------------------------------
# Utilitaire interne
# ---------------------------------------------------------------------------
def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------
def parse(text: str) -> ParsedQuestion:
    """Analyse un texte de QE et renvoie les composants extraits.

    Aucun effet de bord : fonction pure. La couche appelante décide quoi
    faire du résultat (persister en base, embedder, filtrer, etc.).
    """
    if not text or not text.strip():
        return ParsedQuestion(False, None, None, None, None)

    # 1. Détection rappel — on cherche dans TOUT le texte pour ne pas
    # louper une relance qui commencerait par un court préambule (date,
    # référence). Les rappels sont eux-mêmes courts, donc le coût est
    # négligeable.
    is_rappel = bool(RE_RAPPEL.search(text))

    # 2. Contexte
    contexte: str | None = None
    opener_label: str | None = None
    head = text[:OPENER_HEAD_CHARS]
    for label, pat in OPENERS:
        m = pat.search(head)
        if m:
            contexte = _clean(m.group("contexte"))
            opener_label = label
            break

    # 3. Question réelle — cherche l'occurrence la plus précoce dans la queue.
    tail = text[-CLOSER_TAIL_CHARS:]
    tail_offset = len(text) - len(tail)
    earliest_start: int | None = None
    closer_label: str | None = None
    for label, pat in CLOSERS:
        m = pat.search(tail)
        if m and (earliest_start is None or m.start() < earliest_start):
            earliest_start = m.start()
            closer_label = label

    question: str | None = None
    if earliest_start is not None:
        question = _clean(text[tail_offset + earliest_start:])

    return ParsedQuestion(
        est_rappel=is_rappel,
        contexte_extrait=contexte,
        question_extraite=question,
        opener_label=opener_label,
        closer_label=closer_label,
    )


def parse_many(texts: Iterable[str]) -> list[ParsedQuestion]:
    """Convenience : parse une itérable et renvoie une liste."""
    return [parse(t) for t in texts]
