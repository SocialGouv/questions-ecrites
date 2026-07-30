# Extraction du bureau réel depuis les workflows MIN15

> Rédigé 2026-07-28. Contexte : les modules d'attribution bureau ne
> fonctionnent aujourd'hui que pour DGCS (94 % couverture). Comment
> peut-on récupérer des bureaux réels pour les autres directions
> (DGOS, DGS, DSS, DGEFP, DGT) ?

## TL;DR

Les exports **MIN15** de l'outil interne "Réponses" contiennent le
bureau qui a traité chaque QE, encodé dans le champ `poste_etape`. Un
parseur simple (règle : étape "Pour rédaction" OU "Pour attribution",
premier segment ∈ liste des directions connues, ≥ 3 segments) permet
d'extraire un bureau pour **3 216 (question_id, direction)** — dont
**1 275 DSS, 1 143 DGOS, 378 DGS**. Validé à 95 % sur DGCS
(échantillon aléatoire).

L'extraction va dans une **table dédiée** `question_bureau_extract`
(pas dans `question_attributions`) : sources isolées, réversible,
comparable.

Voir SocialGouv/questions-ecrites#45 pour l'implémentation.

## Ce qu'on a découvert en chemin

### 1. Le "bureau_reel_id" existant vient de 15 sources

Le champ `question_attributions.bureau_reel_id` avait déjà des données :
5 925 pour DGCS, 1 011 pour DSS, 0 pour les autres. On pensait ces
données annotées à la main, direction par direction. En creusant les
sources (`question_attributions.source`), on trouve **15 fichiers Excel
distincts** :

| Fichier | Nb attributions | Direction(s) | Origine réelle |
|---|---:|---|---|
| `QuestionsxBureau-Salomé-enrichi.xlsx` | 5 925 | DGCS | **Annotation manuelle** (Salomé) |
| `QE consolidees - DAC MSO - 2026.xlsx` | 3 510 | DGOS, DSS, DGS, DGCS | Export multi-direction (probablement extract REPONSES) |
| `QE DSS en cours et répondues avril 2026.xlsm` | 1 025 | DSS | Extract REPONSES manuellement traité |
| `7 - DGEFP.xls` | 445 | DGEFP | Extract REPONSES |
| `6 - DGT.xls` | 189 | DGT | Extract REPONSES |
| `8 - DGS.xls` | 68 | DGS | Extract REPONSES |
| `10 - DGOS.xls` | 46 | DGOS | Extract REPONSES |
| `9 - DGCS.xls`, `11 - DSS.xls` … | < 25 chacun | | Extract REPONSES par direction |
| autres petites directions | < 5 | DFAS, DNS, DREES, DRH, HDH | Extract REPONSES |

**Insight majeur** : seule DGCS avait une source réellement "à part"
(annotation Salomé). Tout le reste vient d'exports REPONSES faits
manuellement, direction par direction. **On peut donc mécaniser ces
extractions.**

### 2. Le workflow diffère selon les directions

En regardant où apparaît le bureau dans `poste_etape` selon le type
d'étape :

| Direction | "Pour rédaction" au bureau | "Pour attribution" au bureau |
|---|---:|---:|
| **DGOS** | 884 QE | 880 QE (similaire) |
| **DGS** | 289 QE | 186 QE |
| **DGCS** | 153 QE | — (rare) |
| **DGE** | 238 QE | 241 QE |
| **DSS** | **31 QE** | **1 276 QE** ← workflow différent |

**DSS ne trace pas le bureau au moment de la rédaction** — leur
rédaction se fait au "pool" central `DSS QE ministères sociaux`. Le
bureau est nommé au moment de **l'attribution** (préalable) et du
**visa** (validation postérieure). Sans inclure "Pour attribution"
dans la règle, on rate 99 % du signal bureau DSS.

DGOS/DGCS/DGS au contraire tracent bien à la rédaction.

### 3. Structure de `poste_etape`

Champ texte hétérogène, splittable sur ` - ` :

| Nb segments | Cas typique | Utilisable pour bureau ? |
|---:|---|---|
| 1 | `BDC Santé, familles…`, `DSS QE ministères sociaux` | Non — pool ou dispatch admin |
| 2 | `CAB SFAH - POLE RELATIONS AVEC LE PARLEMENT`, `DDC (BDC) MINISTERES SOCIAUX - POLE DIRECTION` | Non — cabinet ou validation |
| **3** | `DSS - SD2 A - REDACTEURS`, `DGOS - SDP - Sous-direction` | **Oui** — bureau nommé |
| **4** | `DGOS - SDRH4 - Temps de travail - secteur privé et sages-femmes` | **Oui** — bureau + description |
| 5+ | Bureaux avec description longue | Oui — bureau_full concatène |

Interprétation :
- `seg[0]` = direction (DGOS, DSS, DGS, …)
- `seg[1]` = sous-direction (SDRH1, SD2 A, SDP, …)
- `seg[2]` = bureau
- `seg[3..]` = description libre du bureau

Directions à filtrer explicitement (**incluses** dans la règle) :
DGCS, DGOS, DSS, DGS, DGEFP, DGT, DFAS, DGE, DGCCRF, DGALN, DGPR,
DGAMPA, DAJ, DRH, DNS, DREES, DIPLP, DARES.

Directions à **exclure** (pas de bureau utile) :
- `BDC XXX` → dispatch admin (pool des correspondants)
- `CAB XXX` / `CABINET` → politique
- `SGG` → point de passage obligatoire (Premier Ministre)
- `DDC` → validation avant transmission

## La règle finale d'extraction

Pour chaque QE et chaque direction observée, on garde l'étape **la plus
récente** vérifiant :

```
type_etape ∈ {'Pour rédaction', 'Pour rédaction interfacée', 'Pour attribution'}
AND poste_etape a >= 3 segments splittés sur ' - '
AND premier segment ∈ KNOWN_DIRECTIONS (18 acronymes)
```

Groupé par `(question_id, direction)` pour capturer les QE
réattribuées entre plusieurs directions (65 % du corpus MIN15).

## Volumes obtenus

Table `question_bureau_extract` après extraction :

| Direction | QE avec bureau extractible | `bureau_reel_id` existant | Gain |
|---|---:|---:|---:|
| DSS | **1 275** | 1 011 (via Excel REPONSES) | +264 (~26 %) |
| DGOS | **1 143** | 0 | **+1 143 (pur gain)** |
| DGS | **378** | 0 | **+378 (pur gain)** |
| DGE | 251 | 0 (hors périmètre social) | +251 |
| DGCS | 153 | 5 925 (via Salomé) | comparable — validation set |
| DGCCRF | 15 | 0 | +15 |
| DAJ | 1 | 0 | +1 |
| **Total** | **3 216 pairs** | | **~1 800 QE en gain net** |

## Validation

Sur les 59 QE DGCS présentes dans les deux sources (Salomé + MIN15),
échantillon aléatoire de 20 QE : **19/20 concordent (~95 %)**.

Exemples :

| QE | `bureau_reel_id` (Salomé) | MIN15 extract | ✓ / ✗ |
|---|---|---|---|
| AN-17-QE-9548 | `[SD3/3B] Insertion, citoyenneté et parcours de vie des personnes en situation de handicap` | `SD3 / Bureau 3B` | ✓ |
| SENAT-17-QE-5411 | `[SD2/2B] Protection de l'enfance et de l'adolescence` | `SD2 / Bureau 2B` | ✓ |
| AN-17-QE-6333 | `[SD4/4B] Emploi et politique salariale` | `SD4 / Bureau 4B` | ✓ |
| AN-17-QE-9036 | `[SD2/2B] Protection de l'enfance et de l'adolescence` | `SD2 / MAJ` | ✗ (rôle admin "Mise à jour", pas un vrai bureau) |

Le seul cas d'écart s'explique par un rôle intermédiaire ("MAJ" =
mise à jour, probablement un contrôleur ou éditeur), à filtrer plus
tard côté enrichissement.

## Utiliser MIN15 comme jeu de test indépendant

Idée méthodo : `question_bureau_extract` n'a jamais été vue par l'algo
d'attribution (le kNN s'entraîne sur `question_attributions`). C'est
donc un **test set réellement indépendant** — bien mieux que le
leave-one-out sur les données d'entraînement, qui sous-estime
systématiquement l'erreur (les QE thématiquement voisines s'aident les
unes les autres).

### Résultat de l'éval indépendante

Comparaison direction top-1 de `questions.direction_algo_id` (cache
kNN existant) vs `question_bureau_extract.direction_txt` :

| Périmètre | QE testées | Top-1 correct | % |
|---|---:|---:|---:|
| **Filtré** (directions présentes en volume dans le training : DGCS, DSS, DGOS, DGS, DGEFP, DGT) | 2 794 | 2 258 | **80.8 %** |
| Non filtré (avec DGE, DGCCRF absentes du training) | 3 046 | 2 258 | 74.1 % |

**Détail par direction (périmètre filtré)** :

| Direction MIN15 | QE testées | Match top-1 | % | Signal training |
|---|---:|---:|---:|---|
| DGCS | 153 | 147 | **96.1 %** ✅ | 5 925 exemples avec bureau |
| DSS | 1 275 | 1 032 | **80.9 %** ✓ | 1 011 avec bureau |
| DGOS | 1 143 | 820 | **71.7 %** ⚠️ | 1 415 direction-only |
| DGS | 378 | 259 | **68.5 %** ⚠️ | 807 attributions |
| DGE (hors périmètre) | 199 | 0 | **0 %** ❌ | jamais dans training |
| DGCCRF (hors périmètre) | 13 | 0 | **0 %** ❌ | jamais dans training |

### Interprétations

1. **L'algo se comporte comme la théorie kNN le prédit** : excellent
   là où il a beaucoup d'exemples humains (DGCS 96 %), bon avec un
   peu (DSS 81 %), moyen avec peu et sans bureau (DGOS 72 %), et
   littéralement nul sans exemple (DGE, DGCCRF à 0 %).

2. **Les 90.4 % top-1 annoncés dans le rapport initial étaient
   optimistes** : le leave-one-out sur les données d'entraînement
   sous-estime l'erreur (les QE thématiquement voisines dans le
   training set s'aident les unes les autres). Sur un jeu réellement
   indépendant, la perf tombe à **~81 %** — soit **~10 pts de biais
   d'optimisme** à corriger dans la communication future.

3. **L'algo est très inégal selon la direction** — 96 % DGCS vs 69 %
   DGS. La moyenne cache un écart de 27 pts. Toute annonce agrégée
   doit être accompagnée du breakdown par direction.

### Ce que MIN15 pourrait apporter comme training

Injecter `question_bureau_extract` dans le training du kNN :

| Direction | Training actuel (attribs) | + MIN15 | Attendu top-1 après |
|---|---:|---:|---|
| DGOS | 1 415 (dir only) | + 1 143 (avec bureau) | 72 % → 82-85 % (estimé) |
| DGS | 807 | + 378 | 69 % → 78-82 % (estimé) |
| DSS | 2 064 | + 1 275 (redondant, mais dense) | 81 % → 83-85 % (marginal) |
| DGCS | 6 326 | + 153 (redondant) | 96 % → 96 % (aucune marge) |

**Levier réel = DGOS et DGS** (+10 à +15 pts top-1 probable).

## Éval A/B : intégrer MIN15 dans le training du kNN

Deux évaluations complémentaires, avec un jeu de test = `question_bureau_extract` (n'a jamais été vu par l'algo) :

### Direction (`scripts/eval_direction_with_min15.py`)

Sur 523 requêtes, gain quasi-nul — MIN15 confirme des directions déjà connues par l'algo, pas d'apport :

| Direction | n | baseline | enriched | Delta |
|---|---:|---:|---:|---:|
| DSS | 227 | 89.0 % | 89.0 % | 0 |
| DGOS | 173 | 72.3 % | 76.9 % | +4.6 |
| DGS | 78 | 74.4 % | 74.4 % | 0 |
| DGCS | 45 | 100 % | 100 % | 0 |
| **TOTAL** | 523 | **82.2 %** | **83.7 %** | **+1.5** |

Conclusion : la direction n'a pas besoin de MIN15 — elle est déjà couverte via `direction_reelle_id`. Le seul mouvement (DGOS +4.6 pts) vient de QE MIN15 qui n'avaient pas d'entrée dans `question_attributions`.

### Bureau (`scripts/eval_bureau_with_min15.py`) — LE gain massif

Sur 526 requêtes, comparaison sur clé bureau canonique (SD/bureau_code) :

| Direction | n | baseline | enriched | Delta |
|---|---:|---:|---:|---:|
| **DSS** | 212 | **1.9 %** | **70.8 %** | **+69 pts** |
| **DGS** | 70 | 1.4 % | **60.0 %** | **+59 pts** |
| **DGE** | 57 | 0 % | **54.4 %** | **+54 pts** |
| **DGOS** | 150 | 0.7 % | **35.3 %** | **+35 pts** |
| DGCS | 36 | 88.9 % | 88.9 % | 0 |
| **TOTAL** | 526 | **7.2 %** | **58.6 %** | **+51 pts** |

Conclusion : injecter MIN15 dans le training kNN débloque littéralement l'algo bureau pour DGOS/DGS/DSS/DGE. Passage de "inutilisable" à "vraiment utile". DGCS reste stable (déjà à 89 %, aucune marge).

**Ordre d'implémentation naturel** :
1. Créer une vue Postgres `question_attributions_all` = UNION(question_attributions avec bureau, question_bureau_extract → canonical key)
2. Modifier le kNN prod (`src/lib/direction/attributionAlgo.ts` ou équivalent bureau) pour lire cette vue
3. Rejouer l'éval fine avant deploy — s'assurer que DGCS reste à 89 % et pas de dégradation ailleurs

## Ce qu'il reste à faire (hors PR #45)

1. **Enrichir le référentiel `bureaux`** avec les bureaux DGOS / DGS /
   DSS observés — actuellement DGCS-first (SD/bureau).
2. **Résoudre `question_bureau_extract` → `bureaux.id`** — mapper les
   libellés texte vers des FKs propres.
3. **Décider si on alimente `question_attributions.bureau_reel_id`
   depuis MIN15** quand vide (aujourd'hui la table extract est
   consommable telle quelle par les modules attribution).
4. **Filtrer les rôles admin** ("MAJ", "Chef de bureau" nu, etc.) qui
   ne sont pas de vrais bureaux métier.
5. **Étendre à d'autres ministères** — la même règle marchera sur
   n'importe quel export MIN15 similaire (Transition écologique,
   Industrie, etc. sont déjà dans nos données mais on n'a pas de
   référentiel bureaux pour eux).

## Pipeline recommandé

Le script `scripts/extract_bureau_from_min15.py` peut être rejoué à
chaque nouvel export MIN15 :

```bash
poetry run python scripts/extract_bureau_from_min15.py --reset
```

Idempotent (upsert par `(question_id, direction_txt)`), rapide (~4s
pour 3 216 pairs), auditable (provenance conservée via
`source_etape_id`).

## Signalé, non résolu

- **Les 5 lignes DFAS + 2 DARES + 2 DRH + 1 HDH** dans
  `question_attributions.source` sont probablement des exports
  REPONSES par direction. On peut les remplacer entièrement par le
  pipeline MIN15 une fois validé.
- **Le fichier `QE consolidees - DAC MSO - 2026.xlsx`** (3 510
  attributions, 0 bureau) sert probablement à peupler
  `direction_reelle_id` seulement. À vérifier avec Victor / DGCS s'il
  a un rôle particulier au-delà.

---

Voir aussi :
- [docs/sources_allotissement.md](sources_allotissement.md) — audit
  parallèle pour les sources d'allotissement (pas d'attribution
  bureau, mais mêmes principes de sources multiples)
- [scripts/extract_bureau_from_min15.py](../scripts/extract_bureau_from_min15.py) — implémentation
- [alembic/versions/c1d2e3f4a5b6_add_question_bureau_extract.py](../alembic/versions/c1d2e3f4a5b6_add_question_bureau_extract.py) — schéma
