# Plan de merge — session 2026-07-17

> Snapshot pour Victor (ou toute personne qui reprend le contexte).
> Obsolète dès que tous les PRs listés sont mergés.

## Vue d'ensemble

6 PRs ouvertes sur 2 repos, issues d'une même session de travail :
- Restauration de la ground truth allotissement historique (bug d'ingestion)
- Décomposition des questions JO en contexte / question / rappel
- Infrastructure d'A/B testing des embeddings + documentation performance

## Ordre de merge

| # | Ordre | Repo | PR | Objet |
|---|:-:|---|---|---|
| 1 | **1er** | qe-front | [#102](https://github.com/SocialGouv/qe-front/pull/102) | Migration Drizzle : 4 colonnes (contexte_extrait, question_extraite, est_rappel, analyzed_at) |
| 2 | **1er (indép.)** | questions-ecrites | [#43](https://github.com/SocialGouv/questions-ecrites/pull/43) | Fix ingestion : `reponse_id = AN-<date>-<page>` pour restaurer les vrais allotissements |
| 3 | 2e | questions-ecrites | [#41](https://github.com/SocialGouv/questions-ecrites/pull/41) | Analyseur régex Python qui peuple les 4 colonnes |
| 4 | 3e | qe-front | [#103](https://github.com/SocialGouv/qe-front/pull/103) | UI : affichage Contexte/Question, callout rappel, redirection silencieuse |
| 5 | 4e | questions-ecrites | [#42](https://github.com/SocialGouv/questions-ecrites/pull/42) | Infrastructure A/B testing + scripts d'eval + document performance v2 |
| 6 | indép. | questions-ecrites | [#40](https://github.com/SocialGouv/questions-ecrites/pull/40) | Fix objet XVII (ancien, non lié à la session) |

## Dépendances techniques

```
qe-front #102 (colonnes DB) ─┬─▶ questions-ecrites #41 (parser) ─┬─▶ qe-front #103 (UI)
                             │                                    │
                             │                                    └─▶ questions-ecrites #42 (eval + docs)
                             │                                              ▲
questions-ecrites #43 (fix ingestion, indépendant) ───────────────────────┘
                                                              nécessaire pour
                                                              que la GT restaurée
                                                              soit correcte à l'eval

questions-ecrites #40 (fix objet XVII) ─── indépendant ─── mergeable à tout moment
```

## Contexte par PR

### qe-front #102 — colonnes DB
Migration Drizzle qui ajoute `contexte_extrait TEXT`, `question_extraite TEXT`, `est_rappel BOOLEAN`, `analyzed_at TIMESTAMPTZ` à la table `questions`. Sans ça les 3 PRs qui suivent plantent en essayant de SELECT sur des colonnes inexistantes.

### questions-ecrites #43 — fix ingestion (indépendant)
Un bug antérieur avait fait `reponse_id = qid` (unique par question) au lieu de `AN-<date>-<page>` (vrai regroupement par page JO). Résultat : les groupes d'allotissement historiques AN 14/15/16 étaient invisibles au SQL (138 105 réponses individuelles au lieu de ~14 000 groupes de vrais allotissements). Le fix restaure le bon format d'ID pour toute nouvelle ingestion.

**Post-merge** : ré-ingérer AN legs 14/15/16 en prod avec le code corrigé. Command : `python scripts/ingest_an_legacy.py --dir data/an_archives/ --skip-embed` (les archives peuvent être re-téléchargées via `scripts/download_an_legacy.py`).

### questions-ecrites #41 — parser
Module Python pur qui extrait le contexte et la question d'une QE JO via regex. Peuple les 4 colonnes créées par #102. 20 tests unitaires. Détaille aussi la détection des rappels administratifs.

### qe-front #103 — UI
Affichage Contexte/Question dans les 3 modules (allotissement, EDR, attribution). Callout distinct pour les rappels. Redirection silencieuse : quand l'ID saisi est un rappel, l'algo travaille sur la question originale citée. Nécessite #102 (colonnes) et #41 (données populées).

### questions-ecrites #42 — infra A/B + docs
Scripts d'évaluation reproductibles :
- `eval_allotissement.py` — hit@20 sur pool retrieve+rerank
- `eval_attribution_kNN.py` — top-1/top-3 leave-one-out
- `embed_questions.py` étendu : `--text-source`, `--variant-tag`, `--only-gt`, `--embedding-provider`
- `compare_variants.py` — récap markdown

Nouvelle table `vec_questions_experiments` pour cohabiter plusieurs variantes A/B sans écraser la prod.

Documentation dans `docs/rapport_performance_v2.md` — trace complète du diagnostic et des chiffres actuels.

### questions-ecrites #40 — fix objet XVII (ancien)
Sans lien avec le chantier du jour. Fix d'extraction de l'objet et de la rubrique pour la legs 17. Mergeable à tout moment.

## Post-merge : étapes nécessaires

1. **Vérifier la migration Drizzle** est bien appliquée en prod (colonnes présentes)
2. **Ré-ingérer AN 14/15/16** avec le code corrigé (#43) pour restaurer les vrais allotissements historiques
3. **Lancer `python scripts/analyze_questions.py --backfill --commit`** pour peupler les colonnes contexte/question/rappel sur tout le corpus
4. **Éventuellement rafraîchir le cache** `question_similar_suggestions` si on veut que les métriques allotissement reflètent la nouvelle base

## Chiffres actuels et à venir

Documentés dans `rapport_performance_v2.md`. Résumé :
- Attribution direction : top-1 = 90,4%, top-3 = 98,5% (baseline reproduit)
- Attribution bureau : top-1 = 83,3%, top-3 = 95,5% (baseline reproduit)
- Allotissement hit@20 sur AN leg 17 (seule GT rigoureuse actuelle) : **54%**

Après re-ingestion AN 14/15/16 (post-merge de #43), on aura ~10 000+ groupes d'allotissements réels supplémentaires — nouveau benchmark plus large et plus solide.
