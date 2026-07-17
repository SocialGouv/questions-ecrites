# Performance des trois modules — v2 (WIP)

> **Document vivant** — mis à jour au fil des découvertes. Successeur du
> `rapport_performance_modeles.pdf` de mai 2026.

## Résumé exécutif *(à consolider)*

Le rapport initial annonçait :

| Module | Métrique | Rapport initial |
|---|---|---:|
| Attribution Direction | top-1 / top-3 | 90,4 % / 98,5 % |
| Attribution Bureau | top-1 / top-3 | 83,6 % / 95,6 % |
| Allotissement / EDR | hit@20 | 74,4 % |

Les mesures ci-dessus sont **reproductibles** avec la méthodologie
d'origine sur la base actuelle, aux nuances près décrites plus bas.

## Ce qui a été refait

### 1. Restauration de la vérité terrain (ground truth)

Un bug d'ingestion antérieur avait donné à chaque question historique
AN legs 14/15/16 un identifiant de réponse unique-par-question
(`AN-LEGACY-<qid>`), même quand plusieurs questions avaient été
répondues avec le même texte (= allotissement historique).

**Symptôme** : le `GROUP BY reponse_id` remontait 1 838 groupes (leg 17
uniquement) au lieu des ~14 000 groupes attendus.

**Fix** : `scripts/dedupe_legacy_reponses.py` regroupe les LEGACY par
hash MD5 du texte de réponse. 12 765 groupes historiques restaurés,
sans altérer les leg 17 modernes.

État de la base :

| Moment | Groupes visibles (≥ 2 questions) |
|---|---:|
| Rapport de mai 2026 | 13 444 |
| Après le fix bug (avant restauration) | 1 838 |
| **Après restauration (aujourd'hui)** | **14 603** |

### 2. Mesures reproduites

`scripts/eval_attribution_kNN.py` reproduit exactement le pipeline
production (leave-one-out kNN pondéré par similarité cosinus) :

| Module | Rapport initial | Reproduit aujourd'hui | Écart |
|---|---:|---:|---:|
| Attribution Direction top-1 | 90,4 % | 90,4 % | 0 |
| Attribution Direction top-3 | 98,5 % | 98,6 % | +0,1 |
| Attribution Bureau top-1 | 83,6 % | 83,3 % | −0,3 |
| Attribution Bureau top-3 | 95,6 % | 95,5 % | −0,1 |

Sur allotissement, mesuré via le cache existant + GT restaurée :

- **hit@20 global (corpus entier)** : **92,7 %** sur 70 358 queries
- **hit@20 par sous-corpus** :

| Sous-corpus | Queries | hit@20 |
|---|---:|---:|
| AN legs 14–16 (LEGACY restaurés) | 66 249 | 95,1 % |
| AN leg 17 (allotissements modernes) | 4 109 | 54,1 % |

Le chiffre global (92,7 %) est **arithmétiquement dominé par les
groupes LEGACY** qui représentent 94 % du poids statistique.

## La qualité de la ground truth : limitation actuelle

La déduplication actuelle des LEGACY par hash MD5 du texte de réponse
crée des groupes **hétérogènes** :

| | Valeur |
|---|---:|
| Écart moyen entre dates de publication | 82 jours |
| Écart max | 1 526 jours (≈ 4 ans) |

**Contrairement à ce que je pensais initialement, un écart temporel
long n'invalide PAS un allotissement** — les ministères peuvent mettre
des mois voire des années à répondre à une question, et regroupent
plusieurs questions posées à des moments différents dans une même
réponse. C'est exactement l'utilité métier du tool.

**Le vrai problème** : sans la métadonnée `page_reponse_jo` (perdue à
l'ingestion des LEGACY), on **ne peut pas distinguer** :
- Un vrai allotissement (N questions → 1 réponse publiée sur la même
  page JO)
- Un template de réponse (N questions → 1 même texte réutilisé mais
  publié séparément à chaque fois)

Le seul sous-corpus où la distinction est faite proprement = **AN leg
17**, ingérée avec le format moderne `AN-YYYYMMDD-page`. C'est aussi le
seul jugé rigoureusement pour l'instant → hit@20 = 54 %.

**La bonne solution** : ré-ingérer XIV, XV, XVI depuis les archives
opendata de l'Assemblée Nationale en préservant `date + page` de la
réponse JO. Le fix code est fait, la migration en cours (voir
post-scriptum).

## Le point clé : que mesure vraiment ce chiffre ?

### Deux mondes distincts dans la GT

- **Legs 14/15/16** : les groupes proviennent d'un artefact — la
  déduplication par texte de réponse identique. En pratique, les
  questions concernées sont des **quasi-copies textuelles** ("templates
  de question" recopiés par plusieurs députés du même groupe politique).
  Retrouver un sibling revient à faire du matching lexical, trivial.

- **Leg 17** : les groupes proviennent de vraies décisions
  administratives — le ministère groupe des questions
  **thématiquement liées** mais rédigées par des députés différents
  avec des angles distincts, des longueurs variées, des chiffres
  cités qui divergent. Le matching est **sémantique** et exigeant.

### Ce que ça implique

1. **Les 74 % / 92,7 % surestiment la performance en usage réel** — les
   cas triviaux LEGACY (questions recopiées + templates de réponse
   réutilisés) sont un artefact de la GT, pas de vrais tests. Ils
   n'existent plus dans la production quotidienne.

2. **Le seul chiffre à annoncer honnêtement** = **54 % hit@20 sur AN
   leg 17**. Sur 20 propositions, au moins une est un vrai sibling…
   dans à peine plus d'un cas sur deux. **C'est mauvais**.

3. **L'algo reste on-topic** — le top-20 leg 17 contient très
   majoritairement des questions du même sujet (100 % on-topic dans le
   cas TUC inspecté). Le manque n'est pas thématique mais sur le
   **matching précis** des groupements historiques.

4. **Pour l'utilisateur** — 10 fois sur 20, l'agent voit zéro vraie
   suggestion pertinente parmi 20 propositions. Perte de confiance
   attendue. Explique le retour utilisateur « c'est nul ».

## Facteurs qui rendent leg 17 objectivement plus difficile

| | AN 14 | AN 15 | AN 16 | AN 17 |
|---|---:|---:|---:|---:|
| Taille moyenne du groupe | 5,8 | 4,5 | 4,0 | **2,8** |
| Taille max | 291 | 92 | 44 | 54 |
| Longueur moyenne texte question | 1 128 | 1 603 | 1 847 | **2 069** |

Trois effets cumulatifs :

1. **Groupes plus petits** → « au moins 1 dans top-20 » mécaniquement
   plus dur avec 2-3 siblings qu'avec 5-6.
2. **Questions plus longues et détaillées** → plus d'espace pour
   diverger dans l'embedding, moins de mots-clés partagés.
3. **Rédaction plus individualisée** → le signal thématique se
   trouve dans l'intersection lexicale, plus mince.

## A/B testing en cours

### Hypothèse initiale (issue des tests utilisateurs de mai 2026)

Les agents rapportaient : « le tool propose des questions sur du
contexte flou, alors qu'on veut répondre à la question ».
Hypothèse : embedder sur `question_extraite` (la vraie demande, sans
préambule ni contexte) devrait améliorer la pertinence.

### Résultats pilote leg 17 (partiel)

Sur leg 17 uniquement (5 093 queries), avec pipeline complet
retrieve+rerank Albert :

| Variante | hit@20 | recall@20 |
|---|---:|---:|
| baseline (texte_question) | **47,8 %** | 33,1 % |
| q_only (question_extraite) | **36,1 %** | 21,2 % |

Sur attribution (partiel leg 17, ~6k queries) :

| Métrique | Baseline | q_only | Écart |
|---|---:|---:|---:|
| Direction top-1 | 90,4 % | 70,6 % | **−20 pts** |
| Direction top-3 | 98,6 % | 93,4 % | −5 pts |
| Bureau top-1 | 83,3 % | 51,9 % | **−31 pts** |
| Bureau top-3 | 95,5 % | 72,8 % | −23 pts |

**Verdict provisoire** : `question_extraite` **dégrade** les trois
modules. La raison probable identifiée sur cas concret : la clôture
extraite est souvent une formule administrative générique
(« il lui demande quelles mesures elle entend prendre pour…»), le
vrai vocabulaire spécifique et discriminant se trouve **dans le corps
de la question** (mots techniques, noms propres, chiffres) — pas dans
la clôture.

### Batchs en cours *(mise à jour à la fin)*

- `q_only` sur les 81 308 questions GT (allotissement + attribution),
  toutes législatures
- `contexte_only` sur les mêmes 81 308 questions (préambule + corps,
  sans clôture)

Objectif : mesures complètes des deux variantes sur GT restaurée, par
sous-corpus. Attendu ~30-45 min.

## Ce que dit le time-anchoring

Objection légitime : mesurer l'algo sur des sibling qui n'existaient
pas au moment de l'allotissement historique fausse le score. Fix :
restreindre le pool aux questions dont `date_publication_jo <=
date_reponse_jo` du groupe.

Résultat sur leg 17 (baseline, sans rerank) :

| Config | hit@20 |
|---|---:|
| Sans time-anchor | 49,9 % |
| **Avec time-anchor** | **50,2 %** |

**Écart marginal (+0,3 pt)**. Le time-anchoring ne résout pas la
sous-performance apparente : l'algo peine à distinguer, parmi les
questions déjà posées à l'époque sur le même sujet, celles qui vont
être groupées ensemble par la décision administrative.

## Pistes d'amélioration identifiées

1. **Fine-tuning du modèle d'embedding** sur les allotissements
   historiques (14 603 groupes = signal supervisé naturel)
2. **Meilleur rerank** — le rerank Albert actuel n'apporte quasi rien
   (parfois même dégrade de 2 pts)
3. **Filtres métier** — restreindre par ministère attributaire, fenêtre
   temporelle glissante (~3-6 mois), etc.
4. **Signal complémentaire** — combiner cosine + heuristiques métier
   (même auteur, même circonscription, même thématique JO)

## Ce qu'il faudra confirmer

- [ ] Résultats `q_only` sur legs 14/15/16 (batch en cours) — attendu :
      bonne perf grâce aux textes quasi-identiques, ce qui validerait
      que la dégradation leg 17 est structurelle (rédactions diverses)
      et non un défaut d'extraction
- [ ] Résultats `contexte_only` — hypothèse : gain sur allotissement
      car garde tout le vocabulaire spécifique sans la formule finale
- [ ] Validation qualitative avec les agents sur des cas concrets

---

## Post-scriptum : origine du bug LEGACY et restauration en cours

### Où est le bug d'ingestion

`qe/ingestion_an.py` extrait bien `page_reponse_jo` du XML archive
(ligne 481-487), mais **ne l'utilise pas** pour construire le
`reponse_id`. Le code force `reponse_id = qid` (l'ID de la question)
avec ce commentaire :

> The date+page-based scheme (AN-YYYYMMDD-page) grouped questions by
> JO publication date alone, creating false allotissement clusters.

Le fix précédent avait supprimé le regroupement à cause de faux
positifs quand on groupait par DATE SEULE (deux questions publiées le
même jour sur des sujets différents). Mais on aurait dû faire
**date + page** — qui identifie précisément une même page du JO, donc
un vrai allotissement. Le code a été corrigé « trop loin » : plus
aucun regroupement.

### Fix appliqué

```python
if texte_reponse:
    if date_reponse and page_reponse_jo:
        reponse_id = f"AN-{date}-{page_reponse_jo}"    # vrai allotissement
    else:
        reponse_id = qid                                # fallback
```

Le format `AN-YYYYMMDD-page` est exactement celui déjà utilisé par
leg 17 (l'ingestion moderne). Il produit des vrais allotissements
vérifiables par la page JO commune.

### Migration en cours

1. Purger les 138 105 LEGACY reponses + pointeurs (fait avant, en cours)
2. Ré-ingérer XVI depuis l'archive locale (~40 Mo, ~5 min)
3. Télécharger + ré-ingérer XIV (2012-2017) et XV (2017-2022)
4. Vérifier que la GT restaurée contient de vrais groupes (page JO
   partagée par plusieurs questions)
5. Ré-évaluer allotissement sur cette GT rigoureuse

Après cette restauration, le benchmark porte sur **~4-5 ans** de recul
historique (au lieu du seul leg 17). Les résultats à venir remplacent
tous ceux du présent document.

---

*Document en cours de rédaction. Dernière mise à jour : session en cours
avec Claude Code.*
