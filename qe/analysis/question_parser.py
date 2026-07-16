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
# Chaque connecteur doit pouvoir absorber une contraction avec article
# ("au sujet DU", "à propos DES", "quant AU", "relative AUX"). On garde
# le connecteur lui-même compact et on liste les formes contractées.
_CONNECTORS = (
    r"(?:"
    r"sur|concernant"
    r"|au\s+sujet\s+d(?:e|u|es|['’])"
    r"|à\s+propos\s+d(?:e|u|es|['’])"
    r"|quant\s+(?:à|au|aux)"
    r"|relative(?:ment)?\s+(?:à|au|aux)"
    r"|relatifs?\s+(?:à|au|aux)"
    r")"
)

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
# Pronom objet optionnel entre le sujet et le verbe :
#   "il lui demande", "il le prie", "il la remercie", "elle l'interroge",
#   "il se demande", etc. Toléré partout — évite de dupliquer chaque motif.
# NB : les élisions ("l'invite") n'ont pas d'espace après l'apostrophe,
# donc on les traite séparément.
_OBJ = r"(?:(?:lui|se|le|la|leur|les)\s+|l['’])?"

CLOSERS: list[tuple[str, re.Pattern[str]]] = [
    # Verbes de demande — pronom + verbe conjugué
    ("il/elle demande",
     re.compile(rf"\b(?:il|elle)\s+{_OBJ}demande(?:ra|rait)?\b", re.IGNORECASE)),
    ("il/elle souhaite",
     re.compile(r"\b(?:il|elle)\s+souhait(?:e|erait|ait|erais)\b", re.IGNORECASE)),
    ("il/elle voudrait/aimerait/prie",
     re.compile(rf"\b(?:il|elle)\s+{_OBJ}(?:voudrait|aimerait|prie|sollicite)\b", re.IGNORECASE)),
    ("il/elle interroge",
     re.compile(rf"\b(?:il|elle)\s+{_OBJ}interroge\b", re.IGNORECASE)),
    ("il/elle remercie",
     re.compile(rf"\b(?:il|elle)\s+{_OBJ}remercie\b", re.IGNORECASE)),
    ("il/elle propose/attend/invite/appelle",
     re.compile(rf"\b(?:il|elle)\s+{_OBJ}(?:propose|attend|invite|appelle)\b", re.IGNORECASE)),
    # Formes inversées : « souhaite-t-elle », « souhaiterait-il », « demande-t-elle », « prie-t-il »
    # Le `t` de liaison est optionnel : présent quand le verbe finit par une
    # voyelle ("souhaite-t-il"), absent quand il finit déjà par t
    # ("souhaiterait-il", "voudrait-il", "pourrait-il").
    ("inversion souhaite/demande/pourrait -t-il/elle",
     re.compile(
         r"\b(?:demande|souhait(?:e|erait|ait)|voudrait|aimerait|pourrait|prie|entend|envisage|compte|remercie|invite|appelle)"
         r"[-\s](?:t[-\s])?(?:il|elle)\b",
         re.IGNORECASE,
     )),
    # Ouvertures de sous-question tardive
    ("aussi/enfin, …",
     re.compile(r"\b(?:aussi|enfin)[,\s]+(?:qu[ei]|comment|dans|il|elle|le\s+ministre|lui)\b", re.IGNORECASE)),
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


# Certains textes du corpus ont été ingérés avec des séquences d'échappement
# LITTÉRALES au lieu de vrais CR/LF (le texte contient `\` + `r` + `\` + `n`
# comme 4 caractères visibles, pas les codes 13/10). Ces caractères parasites
# tuent les word boundaries des regex : `\bIl` ne matche pas dans `n\rIl`
# parce qu'il n'y a pas de frontière entre `n` (lettre) et `I` (lettre).
# On les convertit en espace avant tout matching.
_ESCAPE_SEQ_RE = re.compile(r"\\[rnt]")

def _normalize(text: str) -> str:
    return _ESCAPE_SEQ_RE.sub(" ", text)


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

    # Normalise les séquences d'échappement littérales du corpus qui, sinon,
    # cassent les word boundaries des regex.
    text = _normalize(text)

    # 1. Détection rappel — on cherche dans TOUT le texte pour ne pas
    # louper une relance qui commencerait par un court préambule (date,
    # référence). Les rappels sont eux-mêmes courts, donc le coût est
    # négligeable.
    is_rappel = bool(RE_RAPPEL.search(text))

    # 2. Ouverture — sert uniquement à produire `opener_label` (stats /
    # debug). Le contexte lui-même part du début du texte, pour ne rien
    # perdre du préambule ("M./Mme X interroge Mme la ministre de …").
    opener_label: str | None = None
    head = text[:OPENER_HEAD_CHARS]
    for label, pat in OPENERS:
        if pat.search(head):
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
    question_abs_start: int | None = None
    if earliest_start is not None:
        question_abs_start = tail_offset + earliest_start
        question = _clean(text[question_abs_start:])

    # 4. Contexte — tout ce qui précède la question, sans crop. Le split
    # entier reconstitue le texte original (aux espaces près) : c'est ce
    # que l'UI affiche comme sections "Contexte" et "Question", sans
    # bouton "voir le texte complet" quand les deux sont peuplés.
    # Si aucun verbe de clôture n'est trouvé, on laisse contexte à None
    # et l'UI se replie sur le texte brut.
    contexte: str | None = None
    if question_abs_start is not None and question_abs_start > 0:
        contexte = _clean(text[:question_abs_start]) or None

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
