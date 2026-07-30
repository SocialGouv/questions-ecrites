# Sources d'allotissement — audit complet

> Rédigé 2026-07-28. Question posée : *"peut-on détecter des formes
> d'allotissements dans l'outil Réponses ? Y a-t-il une fonctionnalité
> faite pour au sein de cet outil ?"*
>
> **TL;DR** : le seul champ **explicite** d'allotissement dans les
> exports qu'on possède est la colonne **Commentaires** de l'Excel
> DGCS (`Lot AN XXXX`). L'outil Réponses (MIN15) exporte 12 colonnes
> de workflow **sans champ lot** — probablement parce que le champ
> existe côté tool mais n'est pas remonté dans l'export. Les proxys
> qu'on peut inférer depuis les étapes de workflow (signature ou
> rédaction partagée) donnent trop de faux positifs pour servir de
> GT.

## Les sources qu'on a inspectées

| Source | Type | Explicit allotment field ? | Vol lots ≥2 | Fiabilité GT |
|---|---|:---:|---:|---|
| **JO XML AN opendata** | Import public | Non (mais déductible) | **13 506** | ✅ Haute (validé LLM 98 %) |
| **DGCS Excel** — colonne `Commentaires` | Annotation manuelle DGCS | **Oui** (`Lot AN XXXX`) | **74** | ✅ Haute (source de vérité DGCS) |
| **DGCS Excel** — colonne `Objet` | Libellé thématique DGCS | Non | 175 | ⚠️ Basse (classement thématique, pas allotment) |
| **Outil Réponses (MIN15)** — étape `Pour signature` partagée | Inférence sur workflow | Non | 106 | ❌ Basse (3 % correspondance JO) |
| **Outil Réponses (MIN15)** — étape `Pour rédaction` partagée | Inférence sur workflow | Non | 1 367 | ❌ Basse (4 % correspondance JO) |

## Détail : l'outil Réponses (MIN15)

### Ce qu'on importe

Le script `scripts/import-reponses-extract.py` importe des fichiers
`MIN15_*.xls` (XML SpreadsheetML) exportés depuis l'outil interne du
ministère. Chaque ligne = **une étape de workflow** pour une QE.

**12 colonnes exposées dans l'export** (aucune n'est un identifiant
d'allotissement) :

| # | Colonne | Type | Notes |
|---:|---|---|---|
| 0 | Date JO question | date | Publication au JO de la QE |
| 1 | Numéro | text | 15650, 287, etc. |
| 2 | Parlement | text | AN / SENAT |
| 3 | Parlementaire | text | Nom auteur |
| 4 | Ministère attributaire | text | SGG attribution |
| 5 | Date JO réponse | date | Si répondue |
| 6 | Type étape | text | Pour signature, Pour rédaction, … (17 valeurs distinctes) |
| 7 | Direction étape | text | Direction traitant l'étape |
| 8 | Poste étape | text | Bureau/personne exécutant |
| 9 | Date début étape | date | |
| 10 | Date fin étape | date | |
| 11 | Réponse initiée | bool | Oui/Non |

**Volume actuel** : 9 fichiers, **77 169 étapes**, **4 130 QE
distinctes** (fenêtre 2024-09 → 2026-06, ministère SAS).

### Ce qui n'y est pas

Il n'y a **aucune colonne** qui référence un identifiant de lot, un
groupement, ou une autre QE liée. Les recherches textuelles sur les
champs libres ne trouvent que des faux positifs (`CULOT` = nom propre,
pas `Lot`).

Ce que ça suggère : l'outil interne a probablement un champ ou un
mécanisme d'allotissement (Salomé confirme que les agents raisonnent
en lots), mais **ce champ n'est pas remonté dans l'export MIN15**. Il
faudrait soit demander une évolution de l'export, soit un accès à un
autre export/rapport qui l'inclurait.

### Les proxys qu'on a testés

Deux heuristiques d'inférence, sur la base "même date + même poste" :

**1. Étape `Pour signature` partagée** (2 QE signées le même jour par
la même personne) :

| | Nb groupes |
|---|---:|
| Total groupes (≥2 QE) | 106 |
| Tous membres partagent 1 `reponse_id` JO | **3** (2.8 %) |
| Plusieurs `reponse_id` JO distincts | 100 |
| Aucun encore répondu | 2 |
| Un seul répondu | 1 |

→ Bruit majoritaire. La signature partagée = coïncidence agenda plus
souvent que allotissement.

**2. Étape `Pour rédaction` partagée** (2 QE rédigées ensemble) :

| | Nb groupes |
|---|---:|
| Total groupes (≥2 QE) | 1 367 |
| Tous membres partagent 1 `reponse_id` JO | 58 (4.2 %) |
| Plusieurs `reponse_id` JO distincts | 502 |
| Aucun encore répondu (peut-être allots futurs) | 596 |
| Un seul répondu | 211 |

→ Même conclusion. Les 596 groupes "en cours de rédaction ensemble"
sont **peut-être** de futurs allotments, mais le taux de conversion
sur les groupes finis (58/560 confirmés-vs-non = 10 %) reste faible.

### Verdict outil Réponses

**Non exploitable directement comme source d'allotissement** — pas de
champ explicite, et l'inférence par workflow est trop bruitée pour
servir de GT.

## Détail : le JO XML AN opendata

Source : `qe/ingestion_an.py`. Depuis le fix hash+date :

```python
reponse_id = f"AN-{YYYYMMDD}-{sha1(texte_reponse)[:12]}"
```

Deux QE partagent `reponse_id` **ssi** elles ont reçu exactement le
même texte de réponse dans le même numéro JO. Règle validée contre
les astérisques du PDF officiel du JO (marqueur ministériel de QE
groupées).

**13 506 allotissements identifiés**, **68 741 QE** groupées. Voir
vue Postgres `allotissements_jo` (migration
`e5f6c7d8a9b1_add_allotissements_jo_view`).

Fiabilité cross-validée par 2 LLMs indépendants (Mistral-medium +
gpt-oss-120b) sur les 815 groupes leg 17 = **~98 % vrais allotissements
thématiques**.

**Limite** : le JO ne capture que l'**exécution** ministérielle. Les
cas où DGCS a alloti mais le ministère a fini par répondre séparément
sont invisibles ici.

## Détail : DGCS Excel Commentaires

Fichiers : `1 - TABLEAU QE XVI LEG_Maj *.xlsx` et `XVII LEG_Maj *.xlsx`.
Colonne 17 = `Commentaires`, contenant explicitement des marqueurs
`Lot AN 15650`, `Lot SENAT 14822`, `Lot QE AN 4034`, `Lot Sénat 5029`.

Script d'extraction : `scripts/extract_dgcs_lots.py`. Regex :

```python
LOT_RE = re.compile(
    r"\bLot\s*(?:QE\s+)?(AN|S[EÉ]NAT|SEN)\.?\s*(\d{2,7})\b",
    re.IGNORECASE,
)
```

**74 lots ≥ 2 QE** (leg 16 + 17 combinés). Distribution en cluster
autour de sujets DGCS (Ségur revalorisation, ESAT, EHPAD, etc.).

**Source de vérité pour l'intention d'allotissement DGCS.**

## Croisement DGCS Commentaires vs JO — le résultat clé

Pour chacun des 70 lots DGCS ≥ 2 QE :

| Statut vs JO | Nb | % |
|---|---:|---:|
| ✅ Tous membres partagent 1 `reponse_id` JO (concordant) | 29 | 41 % |
| ⚠️ 2+ `reponse_id` JO distincts (DGCS a alloti, ministère non) | **25** | 36 % |
| Aucun encore répondu au JO | 6 | 9 % |
| Un seul membre répondu à ce jour | 10 | 14 % |

**Interprétation** :

- **29 (41 %)** : les deux sources concordent — vraie allotissement
  exécuté.
- **25 (36 %)** : **DGCS a décidé d'allotir, le ministère a répondu
  séparément**. Ces cas sont **invisibles depuis le JO** — ils
  capturent l'intention DGCS que l'exécution n'a pas suivie. C'est un
  vrai apport de la source DGCS.
- **16 (23 %)** : encore en cours, ne se prononce pas.

Donc DGCS Commentaires n'est **pas plus fiable en volume** que JO
(74 vs 13 506), mais **plus fidèle à ce que veulent les agents** —
c'est leur propre jugement d'allotabilité, avant que le ministère ne
disperse.

## Recommandation pour la ground truth d'éval

Utiliser **les deux sources en complément** :

| Usage | Source à utiliser |
|---|---|
| Éval en volume (mesure statistique) | JO hash+date (13 506 groupes) |
| Éval "agent-aligned" (précision perçue) | DGCS Commentaires (74 lots) |
| Éval combinée (union) | JO ∪ DGCS = 13 531 groupes environ |

À **NE PAS utiliser** :
- `GROUP BY objet` sur Excel DGCS → thématique, pas allotment.
- Étapes MIN15 partagées → bruité, faux positifs > vrais.

## Pistes pour améliorer la couverture des sources

1. **Demander à DGCS un export MIN15 enrichi** — s'il existe un champ
   lot côté outil interne, faire remonter dans l'export.
2. **Ré-annoter systématiquement les Commentaires** — l'annotation
   actuelle est manuelle et incomplète (74 lots pour 4 130 QE
   trackées = 1.8 % taux d'annotation). Une revue des QE
   "annotables" pourrait multiplier ce nombre.
3. **Explorer d'autres directions** — Salomé est DGCS ; les autres
   directions (DGOS, DGS, DSS…) ont probablement des fichiers de
   suivi similaires avec leurs propres conventions d'annotation.
