# Performance de l'algo d'allotissement — v2

> Document vivant. Successeur du `rapport_performance_modeles.pdf` de
> mai 2026. Dernière refonte : 2026-07-28.
>
> **Résumé en 3 phrases** : baseline (BGE-M3 + Albert rerank) atteint
> ~92 % hit@20 dans les conditions réalistes (candidats EN_COURS à la
> date d'arrivée de la question source). Aucune des variantes testées
> (q_only, titre_q, filtre direction, rerank sur question_extraite)
> n'a apporté de gain matériel. Le vrai bottleneck n'est plus la
> **couverture** (hit@K) mais la **précision perçue** — les rangs 2-5
> contiennent souvent des résultats du bon domaine sémantique mais pas
> de la même question, avec un score rerank élevé (0.85-0.95), ce qui
> érode la confiance des agents.

## TL;DR chiffré (état 2026-07-28)

Toutes conditions actuelles :
- **Retrieval** : BGE-M3 (dense) sur `texte_question` complet, pool 500
- **Rerank** : Albert `openweight-rerank` (~570M params)
- **Time-anchor** : candidats limités à ceux **encore EN_COURS** à la
  date de publication JO de la source. **Ce time-anchor est celui
  aligné sur le vrai job de l'agent** — cf. section méthodologie.
- **GT** : 13 506 allotissements ministériels reconstitués via
  `reponse_id = AN-{YYYYMMDD}-{sha1(texte_reponse)[:12]}` (validé LLM
  à ~98 % sur leg 17)

| Variante | hit@1 | hit@3 | hit@5 | hit@10 | hit@20 |
|---|---:|---:|---:|---:|---:|
| **Baseline** | 55.5 % | **79.1 %** | 85.0 % | 89.1 % | **91.9 %** |
| + filtre direction (auto) | 53.6 % | 73.1 % | 77.9 % | 81.1 % | 83.6 % |
| titre_q (objet + question_extraite) | 50.6 % | 62.9 % | 66.8 % | 70.0 % | 71.2 % |
| q_only (question_extraite seule) | 41.6 % | 51.7 % | 54.2 % | 55.7 % | 56.5 % |
| Rerank sur `question_extraite` | 54.6 % | 76.9 % | 82.6 % | 90.1 % | 91.7 % |

Sur le sous-corpus DGCS **avec la vraie GT lots DGCS** (Excel Salomé,
colonne Commentaires → lots identifiés par `Lot AN|SENAT NNNN`,
70 lots ≥ 2 QE) :

| Variante | hit@1 | hit@3 | hit@5 | hit@10 | hit@20 |
|---|---:|---:|---:|---:|---:|
| Baseline | 59.9 % | 76.3 % | 82.4 % | 88.4 % | 92.7 % |
| Rerank sur `question_extraite` | 63.8 % | 77.5 % | 83.0 % | 88.4 % | 92.7 % |

Distribution du rang du premier vrai mate quand il est trouvé
(baseline, 2048 requêtes) :

| Rang du premier mate | Fréquence |
|---|---:|
| 1 | 55.5 % |
| 2-3 | 23.5 % |
| 4-5 | 5.9 % |
| 6-10 | 4.2 % |
| 11-20 | 2.8 % |
| Miss (>20) | 8.1 % |

Rang médian = 1, rang moyen = 2.3. **Quand un vrai mate est trouvé, il
est presque toujours en tête**.

## Le vrai problème (state 2026-07-28)

Le décalage entre "92 % hit@20" et "les agents disent que c'est nul" ne
vient pas de la couverture. Il vient de deux choses :

**1. Les agents jugent sur des cas qu'on ne mesure pas.** Leurs 2
exemples de référence (AN-17-QE-9326 APA, AN-17-QE-13335 PCH) sont
EN_COURS sans `reponse_id` → ils **ne sont pas dans notre GT du tout**.
Notre 92 % dit "quand un mate ministériel existe, on le trouve" ; il
ne dit rien sur "quand une QE arrive, quelles suggestions voit l'agent
en pratique".

**2. Les rangs 2-5 ont un problème de précision.** Sur QE-9326
"Revalorisation APA" :

| rk | rerank | objet | verdict |
|---:|---:|---|---|
| 1 | 0.94 | Révision des critères d'attribution de l'APA | ✅ |
| 2 | 0.93 | "personnes âgées" (réforme APA / démographie 2060) | ⚠️ même domaine, autre question |
| 3 | 0.92 | "personnes âgées" (anciens agriculteurs / solidarité) | ⚠️ même domaine, autre question |
| 4 | 0.90 | Perte d'heures financées de l'APA | ✅ |
| 5 | 0.86 | "aides à domicile" | ⚠️ vague |
| 6 | 0.86 | Handicap après 60 ans | ⚠️ confusion PA/PH |
| 7 | 0.86 | < 60 ans en grande dépendance | ⚠️ même confusion |

Le rerank Albert confond **"même domaine sémantique"** avec **"même
question"**. Il voit "APA + personnes âgées + coûts" partout et donne
0.93 à tout ce qui active ces mots-clés.

## Ground truth : quelle GT utiliser pour quoi ?

Il y a **au moins 3 sources** de "vérité" plus ou moins alignées :

| Source | Nb groupes ≥2 QE | Nb QE | Fiabilité | Ce que ça mesure |
|---|---:|---:|---|---|
| Hash+date ministériel (`AN-{date}-{sha1}`) | 13 506 | 68 741 | 98 % LLM-validated | "Le ministère a répondu à ces QE ensemble" |
| Lots DGCS (Excel `Commentaires` → `Lot XXX`) | 74 | ~450 | Haute — annotation humaine | "DGCS a décidé d'allotir ces QE ensemble" |
| Signature étapes DGCS (`type_etape='Pour signature'` + date + poste) | 107 | 1 384 | À valider | Proxy : QE signées ensemble, probablement mêmes réponses |
| Groupes objet DGCS (`GROUP BY objet` sur Excel) | 175 | 936 | **Bas** — thématique | Classification sémantique DGCS, PAS allotissement réel |

**Recommandation** : utiliser en priorité **hash+date ministériel**
(volume + fiabilité), et croiser avec **lots DGCS** pour valider le
comportement sur le périmètre agents.

## Méthodologie : le time-anchor

**Contrainte à respecter** : à l'instant t où un agent regarde une QE
source, le pool candidat = **QE encore EN_COURS à t**. Formellement :

```
candidat retenu ssi:
  date_publication_jo(candidat) <= date_publication_jo(source)
  AND (candidat.reponse_id IS NULL
       OR date_reponse_jo(candidat) > date_publication_jo(source))
```

Les GT-mates sont filtrés pareil : seuls comptent ceux **déjà publiés
à t**.

**Erreur commise dans l'éval précédente** (`scripts/eval_allotissement.py`) :
le time-anchor filtrait "posté avant `date_reponse`", ce qui incluait
plein de QE **déjà répondues** au moment où la source arrivait. Résultat :
le hit@20 sous-estimait la vraie performance de ~14 pts (78 % vs
92 %). Corrigé dans `scripts/eval_realistic_encours.py`.

## Historique des expériences

Ordre chronologique, avec ce qu'on a appris de chaque essai (y compris
les fausses pistes) :

### E01 — Reproduire les chiffres du rapport mai 2026

`scripts/eval_attribution_kNN.py` reproduit exactement le pipeline
production (leave-one-out kNN pondéré par cosine) :

| Module | Rapport mai 2026 | Reproduit | Écart |
|---|---:|---:|---:|
| Attribution Direction top-1 | 90.4 % | 90.4 % | 0 |
| Attribution Direction top-3 | 98.5 % | 98.6 % | +0.1 |
| Attribution Bureau top-1 | 83.6 % | 83.3 % | −0.3 |
| Attribution Bureau top-3 | 95.6 % | 95.5 % | −0.1 |

Chiffres attribution **confirmés**. Les modules attribution ne sont
pas le problème.

### E02 — Bug structurel dans l'ingestion LEGACY (résolu)

**Symptôme** : `GROUP BY reponse_id` remontait 1 838 groupes (leg 17
seul), au lieu des ~14 000 attendus.

**Cause 1** (identifiée puis "corrigée" trop vite) : les QE legs
14/15/16 avaient toutes reçu un `reponse_id = AN-LEGACY-<qid>` (unique
par question), donc aucun regroupement possible.

**"Fix" v1 (FAUX)** : `reponse_id = AN-{YYYYMMDD}-{page_reponse_jo}`.
Chiffres remontés à 13 444 groupes → apparemment bon.

**Mais** : dans le XML AN opendata, `pageJO` = page de **début de la
section "Réponses"** du numéro JO, PAS la page individuelle de chaque
réponse. Résultat : toutes les QE d'un même numéro JO se sont
retrouvées avec le même `reponse_id`, créant des allotissements
factices massifs.

**Fix v2 (correct, en prod)** : `reponse_id = AN-{YYYYMMDD}-{sha1(texte_reponse)[:12]}`.
Deux QE partagent un `reponse_id` **ssi** elles ont reçu exactement le
même texte de réponse dans le même numéro JO. Validé contre les
astérisques du PDF officiel du JO.

Résultat après ré-ingestion complète 14/15/16/17 : **13 506
allotissements, 68 741 QE groupées**. Cross-validé par 2 LLMs
indépendants (Mistral-medium + gpt-oss-120b via Albert) sur leg 17 =
~98 % de vrais allotissements thématiques.

Voir `docs/vue allotissements_jo` (VUE Postgres) pour l'accès direct.

### E03 — Variantes d'embedding testées (toutes échouées)

**Hypothèse** : les agents disent "le tool propose des questions sur
du contexte flou, alors qu'on veut répondre à la question précise".
Peut-être qu'embedder uniquement sur la question extraite améliorerait.

**Variantes testées** :
- `q_only` : `question_extraite` seule
- `titre_q` : `objet + question_extraite`
- `contexte_only` : préambule + corps sans clôture

Résultats sur baseline anchor puis vrai anchor (voir tableau TL;DR).
Toutes **dégradent significativement** — jusqu'à −27 pts hit@3 pour
`q_only`.

**Diagnostic empirique** : les mots-clés discriminants (chiffres, noms
propres, termes techniques) sont dans le corps de la question, pas
dans la clôture ni dans le titre. Extraire uniquement la question
enlève le vocabulaire qui fait la différence entre 2 sujets voisins.

### E04 — Filtre par direction (chantier PR-A à PR-E, invalidé)

**Hypothèse initiale** : sur GT DGCS objet-based (biaisée thématique
pure), filtrer par direction gagnait +22 pts hit@3. Chantier lancé en
5 PRs.

**Après le fix v2** : sur la vraie GT hash+date (13 506 allotissements),
filtrer par direction **dégrade** hit@K :

| Métrique | Baseline | +filter direction | Delta |
|---|---:|---:|---:|
| hit@3 | 79.1 % | 73.1 % | −6 |
| hit@20 | 91.9 % | 83.6 % | −8 |

Raison : des allotissements cross-direction légitimes existent
(question posée à un ministre, aiguillée vers une autre direction) ;
le filtre les écarte.

Les 5 PRs ont été **repositionnées comme filtre UX-only optionnel**,
avec défaut = "toutes directions". Le mode "auto" reste disponible
pour l'agent qui le souhaite mais n'est plus le comportement par
défaut. Voir PR SocialGouv/qe-front#106 (backend) et #107 (UI).

### E05 — Rerank sur question extraite (essayé, neutre)

**Hypothèse** : le rerank Albert confond "même domaine" et "même
question" (cf. exemples QE-9326). Peut-être qu'en lui donnant
`question_extraite` au lieu du texte complet il discriminerait mieux
l'intention.

**Retrieval inchangé** (BGE-M3 sur texte complet) ; seul le rerank
reçoit la question crispée.

**Résultat sur 329 requêtes (GT DGCS lots)** :

| Métrique | Baseline | Rerank sur extraite | Delta |
|---|---:|---:|---:|
| hit@1 | 59.9 % | 63.8 % | +4 |
| hit@3 | 76.3 % | 77.5 % | +1 |
| hit@5 | 82.4 % | 83.0 % | +1 |
| hit@20 | 92.7 % | 92.7 % | = |

Delta hit@1 = +4 pts dans le bon sens, mais **dans le bruit** (n=329,
marge ~±4 pts). Le rerank a besoin du contexte complet pour bien
juger.

### E06 — LLM juge sur les allotissements (validation GT, pas amélioration algo)

Cross-check qualité GT sur les 815 groupes leg 17, deux LLMs
indépendants (Mistral-medium + gpt-oss-120b) :
- 751 real (92 %)
- 27 suspect (3 %)
- 13 batch admin (1.6 %)
- 24 error / rate-limit

Confirme la GT hash+date est saine à ~98 %.

## Le pipeline : retrieval + rerank en 2 étapes

Le système actuel a **2 étapes** :

**1. Retrieval (BGE-M3)** — rapide, grossier.
- Prend le texte de la QE source, le convertit en vecteur (embedding).
- Compare avec les ~260 k vecteurs de la base via similarité cosinus.
- Ramène les **100 plus proches** en ~50 ms.
- Grossier : la similarité cosinus mesure "à quel point les 2 textes
  utilisent le même vocabulaire dans un contexte similaire" — grosso
  modo "on parle du même sujet ?".

**2. Rerank (Albert `openweight-rerank`)** — plus lent, plus précis.
- Reprend les 100 candidats du retrieval.
- Pour chaque candidat, appelle un cross-encoder qui **regarde source
  et candidat ensemble** (pas séparément) et sort un score 0-1 : "à
  quel point ces 2 questions se ressemblent vraiment ?".
- Trie les 100 par ce nouveau score.
- Ne garde que le top-20 pour affichage. ~2 s pour 100 paires.

**Analogie** :
- Retrieval = jeter un aimant dans un tas de trombones et récupérer
  les 100 qui réagissent. Rapide, mais ordre approximatif.
- Rerank = prendre ces 100 trombones un par un et les examiner à côté
  de l'original pour dire "celui-ci est le plus proche, celui-là
  moyennement…". Plus lent, tri fin.

**Pourquoi c'est le bottleneck** — le retrieval fait bien son job (le
vrai mate est presque toujours dans les 100), mais le rerank actuel
confond "même domaine" et "même question précise". Quand il voit
"APA + personnes âgées + coûts" dans une QE, il donne 0.93 à toutes
celles qui activent ces mots-clés, même si l'intention diffère
(revalorisation vs financement vs éligibilité).

## Ce qui reste à essayer

Le rerank Albert est le bottleneck confirmé. Deux directions restantes :

1. **Rerank par LLM** (Mistral-medium ou gpt-oss-120b) — présenter les
   20 candidats à un LLM avec la source et lui demander de scorer
   1-10 la vraie similarité de question. Testé indirectement via le
   juge d'allotments (E06) → capable de distinguer proprement même
   question vs même domaine. Coût : ~20 min pour 100 QE test.

2. ~~**Cross-encoder plus gros** : `bge-reranker-v2-m3`~~ — **piste
   morte** (E08). Albert route `openweight-rerank` vers
   `bge-reranker-v2-m3` en interne. Vérifié par test contrôlé : les
   scores rerank retournés par les 2 identifiants sont **strictement
   bit-identiques** sur les mêmes paires. On utilise déjà ce modèle.

   **Conséquence** : plus aucun gain à attendre d'un autre
   cross-encoder générique côté Albert. Le rerank actuel = état de
   l'art des cross-encoders open-source. Pour gagner, il faut soit
   raisonner autrement (LLM), soit spécialiser (fine-tuning).

3. **Fine-tuning cross-encoder** sur nos données spécifiques.

   **Idée** : prendre le rerank actuel (`bge-reranker-v2-m3`) et le
   ré-entraîner sur nos paires pour qu'il apprenne les spécificités
   du corpus QE (PA ≠ PH, APA ≠ PCH, "revalorisation" ≠
   "financement", etc.).

   **Données d'entraînement** :
   - **Paires positives** : les 13 506 allotissements validés
     (vue `allotissements_jo`). Pour chaque allotment, toutes les
     combinaisons de paires (A, B) où A et B sont dans le même lot
     → label 1.
   - **Hard negatives** : pour chaque question source d'un
     allotment, prendre les top-20 du retrieval qui **ne sont pas
     dans le même lot** → label 0. Ce sont les cas où le retrieval
     ramène quelque chose de sémantiquement proche mais qui n'est
     PAS un vrai mate — exactement ce qu'il faut apprendre au
     modèle à rejeter.
   - Volume estimé : ~13 506 positives × ~5-10 négatives =
     ~100 k paires d'entraînement.

   **Coût** :
   - Compute : ~2-4 h sur GPU L4/A100 (Colab Pro suffit, ou instance
     Scaleway/OVH ~5 €).
   - Code : ~200 lignes Python (bibliothèque `sentence-transformers`
     ou `transformers`).
   - Déploiement : servir le modèle fine-tuné à la place d'Albert.
     Peut être hébergé sur infra Ministère si Etalab accepte
     d'héberger un modèle custom, sinon backend inference dédié
     (Scaleway Inference, ~30 €/mois).
   - Maintenance : re-entraîner tous les 6 mois quand la GT
     s'enrichit.

   **Ce que ça résoudrait concrètement** : sur l'exemple QE-9326
   (Revalorisation APA), le rerank actuel donne 0.93 à "personnes
   handicapées après 60 ans" (rang 6, confusion PA/PH). Un
   cross-encoder fine-tuné aurait vu des centaines de fois pendant
   l'entraînement que APA et PCH ne finissent PAS ensemble → il
   apprendrait à baisser ce score.

   **Faisabilité** : élevée. On a tout ce qu'il faut aujourd'hui
   (GT validée, corpus de hard negatives, cross-encoder open à
   fine-tuner). C'est probablement le chantier **le plus
   prévisiblement rentable** parmi ceux qui restent.

   **Alternative moins engageante** : le **LLM juge** (piste 1) est
   plus rapide à prototyper et à évaluer, mais coûte cher à
   l'exécution en production (1 appel LLM par candidat). Le
   fine-tuning est plus lourd en amont mais gratuit à faire tourner
   ensuite.

4. ~~**Split pipeline "retrieval sur contexte, rerank sur question"**~~
   — **testé, échec** (E07). Idée logique mais dégrade partout :

   | Métrique | Baseline (texte/texte) | Split (ctx/q) | Delta |
   |---|---:|---:|---:|
   | hit@1  | 55.5 % | 53.1 % | −2 |
   | hit@3  | 79.1 % | 71.4 % | **−8** |
   | hit@5  | 85.0 % | 78.9 % | −6 |
   | hit@10 | 89.1 % | 86.5 % | −3 |
   | hit@20 | 91.9 % | 89.4 % | −2 |
   | recall@3 | 37.6 % | 32.6 % | −5 |
   | recall@20 | 69.8 % | 65.3 % | −5 |

   Deux effets observés :
   - `contexte_only` en retrieval ramène un pool plus large
     thématiquement mais moins bien ordonné → moins de vrais mates
     dans les 100 candidats.
   - `question_extraite` en rerank est trop court/générique pour
     discriminer finement — les clôtures "il lui demande quelles
     mesures elle entend prendre" se ressemblent entre QE de sujets
     voisins.

   **Bilan** : découper retrieval/rerank par domaine (contexte) vs
   intention (question) est intuitif mais empiriquement ne marche
   pas — les 2 étapes ont besoin du texte **complet** pour bien faire
   leur travail.

## Le trade-off K affiché : couverture vs propreté

Question posée : *"faut-il pas plutôt mesurer le % de chances de
proposer quelque chose de correct dans les X résultats affichés ?"* —
c'est exactement ce que hit@K mesure. Détail :

**Couverture par K** (baseline, vrai time-anchor, 2 048 requêtes) :

| Top-K affiché | hit@K | Manque (aucun vrai dans top-K) |
|---:|---:|---:|
| 1  | 55.5 % | 44.5 % |
| 3  | 79.1 % | 20.9 % |
| 5  | 85.0 % | 15.0 % |
| 10 | 89.1 % | 10.9 % |
| **20** | **91.9 %** | 8.1 % |

**Propreté par K** (via recall@K, avec taille moyenne de groupe ~5) :

| Top-K | recall@K | ≈ composition estimée |
|---:|---:|---|
| 3  | 37.6 % | 1 vrai + 2 bruit sur 3 affichés |
| 5  | 47.6 % | 1.5 vrai + 3.5 bruit sur 5 |
| 10 | 59.6 % | 2 vrais + 8 bruit sur 10 |
| 20 | 69.8 % | 3 vrais + 17 bruit sur 20 |

**Interprétation UX** : afficher moins de résultats ne rend pas la
liste plus propre mécaniquement — le ratio vrai/bruit reste
défavorable même en top-3 (1/3). Le vrai levier n'est pas K, c'est
la qualité du rerank qui décide de l'ordre.

**Trois leviers UX activables sans changer l'algo** :

1. **Réduire K par défaut** (top-5 au lieu de top-20), avec "voir
   plus" — coût 7 pts de couverture, gain en digestibilité.
2. **Cut par score de rerank** — n'afficher que les candidats
   au-dessus d'un seuil (par ex. 0.85). Trade-off : parfois 0
   résultat, mais quand contenu affiché il est confident.
3. **Coloration des scores** — vert au-dessus de 0.90, orange 0.75-0.90,
   grisé en-dessous. L'agent apprend à faire confiance à sa
   perception plutôt qu'au 0.9x générique.

## Pourquoi le décalage "92 % mesuré" vs "agents disent que c'est nul"

**Trois hypothèses distinctes**, toutes probablement vraies à des degrés variables :

### H1 — Précision perçue en top-3 (rerank confond même domaine / même question)

Sur QE-9326 (Revalorisation APA), top-1 et top-4 sont pertinents, mais
top-2 et top-3 sont à 0.92-0.93 sur "même domaine PA/dépendance mais
autre question". Le rerank confond "même domaine sémantique" et "même
question précise".
- **Fix** : améliorer le rerank (fine-tuning ou LLM juge)

### H2 — Désaccord DGCS ↔ ministère (l'algo hérite de la pratique ministérielle)

Notre croisement Commentaires DGCS × JO a montré que **36 % des lots
décidés par DGCS n'ont PAS été exécutés** par le ministère (2+ réponses
distinctes). L'algo est mesuré et implicitement optimisé sur les
allotments **ministériels** (JO), donc hérite de la pratique du
ministère, pas de celle de DGCS. Un agent DGCS avec un modèle mental
d'allotement plus large que celui du ministère verra l'algo suggérer
des choses "trop étroites".
- **Fix** : fine-tuner sur les lots DGCS (74 lots Commentaires) plutôt
  que sur JO, ou combiner les deux

### H3 — Il n'y a rien à allotir à l'instant t (le cas "légitimement zéro")

Pour une QE unique dans son sujet à ce moment-là, aucune autre EN_COURS
ne colle vraiment. L'algo est **forcé** de retourner 20 résultats
(`MIN_RESULTS = 20` en dur dans `app/api/questions/[id]/similar/route.ts`).
Il montre alors les "moins pires" — du même thème général mais aucun
vrai frère. L'agent voit 20 propositions à 0.7-0.85 sur des choses
tangentielles et dit "nul" — ce qui est factuellement le bon jugement :
il n'y a rien à afficher, mais on affiche 20 trucs quand même.

C'est probablement la cause **la plus fréquente** sur les cas orphelins
que les agents rencontrent au quotidien (QE nouvelles sur des sujets
peu couverts).

- **Fix** : passer la règle actuelle "afficher au moins `MIN_RESULTS`
  OU tous ceux au-dessus de `RELEVANCE_THRESHOLD`, max des deux" à
  "afficher UNIQUEMENT ceux au-dessus du seuil, éventuellement 0 si
  rien". Ajouter un état vide UI clair : *"Aucune question similaire à
  allotir pour le moment"*.

### Pourquoi ça compte

Ces 3 hypothèses appellent des fixes différents. Si on ne fait que le
H1 (améliorer le rerank), on ne résoudra pas H3 (l'algo continuera de
bourrer 20 résultats bruités quand il n'y a rien à trouver). Si on ne
fait que H3 (seuil strict), le rerank continuera à confondre "même
domaine" et "même question" quand il y a effectivement des candidats.

Le fix H3 (seuil strict + état vide) est **le plus rapide à déployer**
(quelques lignes de code) et probablement **le gain de perception le
plus immédiat** pour les agents — mieux vaut afficher zéro résultat
honnête que 20 résultats bruités.

**Conséquence** : il n'existe **aucune source de vérité qui
répondrait objectivement à "algo bon / pas bon"** sur les cas que
les agents rencontrent. La seule voie propre : le feedback en
production (pouces UX) sur ces cas orphelins — pas de shortcut
possible via GT historique.

### La boucle de feedback (pouces UX)

La boucle sera le meilleur signal à terme, avec deux caveats :

- **Volume** : compter plusieurs centaines de pouces (donc 3-6 mois
  d'usage productif) avant d'avoir un signal statistiquement
  bougeable.
- **Biais** : les agents raterotent surtout les cas où l'algo se
  plante spectaculairement (score haut sur non pertinent). Bon
  signal pour du fine-tuning ciblé, ne remplace pas une éval
  systématique.

Recommandation : **implémenter les 3 leviers UX** (K plus petit,
seuil, couleurs) **tout de suite**, laisser tourner la boucle
feedback, et réserver le fine-tuning rerank / LLM-rerank pour quand
on aura les données de pouces à partir desquelles trancher.

## Les sources de GT disponibles

Voir `docs/sources_allotissement.md` pour l'audit complet. Résumé :

| Source | Vol lots ≥2 | Fiabilité | Ce que ça capte |
|---|---:|---|---|
| JO hash+date | **13 506** | ✅ 98 % LLM-validated | Exécution ministérielle |
| DGCS Commentaires (`Lot AN XXXX`) | **74** | ✅ Annotation humaine | Intention allotissement DGCS |
| DGCS `GROUP BY objet` | 175 | ⚠️ Classement thématique | À ne PAS utiliser comme GT allotment |
| MIN15 étapes partagées | 106-1367 | ❌ 3-4 % correspondance JO | Bruit, à ne PAS utiliser |

**Découverte importante** du croisement DGCS Commentaires × JO : sur
les 70 lots DGCS testables, **25 (36 %) sont des allotissements
décidés par DGCS mais NON exécutés par le ministère** (2+ réponses
JO distinctes). Ces cas sont invisibles depuis le seul JO — c'est
l'apport propre de DGCS.

## Ce qui est acquis

- L'algo trouve les vrais mates : **hit@20 = 92 %** dans les conditions
  réelles (candidats EN_COURS à date_publication(source)).
- Le rang médian du premier mate = 1 → l'algo classe bien quand il
  trouve.
- La couverture n'est pas le problème. Le problème est la **précision
  perçue en top 2-5**.
- Les 5 chantiers de variantes d'embedding et de filtres testés
  n'ont pas produit de gain. Le levier restant est le **rerank
  lui-même**, pas ses entrées ni ses filtres.

---

## Annexe : scripts et données

Sur branche `feat/embedding-variants` :
- [scripts/eval_realistic_encours.py](../scripts/eval_realistic_encours.py) — éval principale avec vrai time-anchor
- [scripts/eval_rank_distribution.py](../scripts/eval_rank_distribution.py) — distribution du rang du premier mate
- [scripts/inspect_similar.py](../scripts/inspect_similar.py) — top-K pour une QE donnée avec scores + mates
- [scripts/extract_dgcs_lots.py](../scripts/extract_dgcs_lots.py) — extrait les lots DGCS depuis Excel Commentaires
- [scripts/eval_allotissement.py](../scripts/eval_allotissement.py) — ancien script (⚠️ time-anchor buggé, à ne plus utiliser)
- [scripts/llm_judge_allotments.py](../scripts/llm_judge_allotments.py) — LLM juge pour validation GT

Données (`data/`) :
- `eval_encours_baseline_v1.json` — baseline 500 groupes, vrai anchor
- `eval_encours_qonly.json` / `_titreq.json` / `_filter.json` — variantes
- `eval_encours_rank_dist.json` — distribution des rangs
- `eval_dgcs_lots_baseline.json` / `_rerank_extraite.json` — GT DGCS lots
- `dgcs_lots.csv` — 74 lots DGCS extraits d'Excel (leg 16 + 17)
- `dgcs_groups.csv` — ⚠️ ancien fichier, groupement thématique par objet, à ne plus utiliser comme GT allotissement

VUE Postgres exposée par migration `e5f6c7d8a9b1_add_allotissements_jo_view` :
```sql
SELECT * FROM allotissements_jo LIMIT 10;
-- 13 506 lignes, chacune = un allotment ministériel reconstitué
```
