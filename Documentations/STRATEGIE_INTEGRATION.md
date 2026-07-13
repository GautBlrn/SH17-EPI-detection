# Stratégie d'intégration IA

Ce document répond à la section 5.1 du cahier des charges (BC02 - Projet 1) : cas d'usage prioritaires, impact métier, feuille de route de déploiement. Il s'appuie sur les résultats réels de l'évaluation du modèle retenu (`yolo11l_full-12`, ré-entraîné en local, cf. `Applications/pipeline_SH17.ipynb`, section 3).

## 1. Cas d'usage prioritaires

### EPI critiques à surveiller en premier

L'évaluation par classe du modèle retenu donne, en mAP@50-95 sur le jeu de test :

| Classe | mAP@50-95 | Lecture |
|---|---|---|
| `head` | 0.730 | Détection fiable, sert de base à la logique de conformité |
| `helmet` | 0.499 | Performance moyenne, classe rare (652 instances en train) mais visuellement distinctive |
| `safety-vest` | 0.307 | Point faible du modèle : rappel insuffisant, ne peut pas fonder une alerte ferme |

Le modèle est par ailleurs conservateur sur la majorité des classes (précision > rappel à seuil égal), à l'exception notable de `safety-vest` où le rappel (0.606) dépasse légèrement la précision (0.587) : la classe reste le point faible du modèle, ce qui renforce l'argument de ne pas en faire une alerte ferme. En contexte sécurité, une alerte manquée (faux négatif) est plus coûteuse qu'une fausse alerte, ce qui pousse à retenir un seuil de confiance plutôt bas côté casque et à ne pas fonder de décision automatique sur le gilet tant que son rappel n'est pas amélioré.

**Priorisation retenue :**
1. **Casque (`helmet`/`head`)** : seul EPI sur lequel repose une alerte ferme ("tête nue"), fondée sur l'association géométrique casque/tête (cf. `helmet_covers_head` dans `Applications/app/app.py`). C'est l'EPI à surveiller en premier, cohérent avec le risque le plus grave (traumatisme crânien).
2. **Gilet réfléchissant (`safety-vest`)** : indice affiché mais non bloquant, en attendant un rappel suffisant (piste d'amélioration : plus de données d'entraînement sur cette classe, ou seuil de confiance dédié plus bas).
3. **Gants, lunettes** : hors périmètre de l'alerte automatique dans la version actuelle (classes présentes dans le dataset et détectées mais non intégrées à la logique de conformité) ; à envisager dans une phase ultérieure une fois le socle casque/gilet validé en conditions réelles.

### Chantiers pilotes

Déploiement recommandé sur un chantier pilote unique, de taille moyenne, avec :
- une zone d'entrée/sortie fixe (portique ou caméra fixe), qui limite les angles de vue et facilite la validation du taux de faux positifs/négatifs par comparaison avec un contrôle humain,
- un responsable sécurité identifié comme interlocuteur pour la validation des alertes et la remontée de terrain,
- une durée de test d'au moins 4 à 6 semaines avant toute décision de généralisation, pour couvrir des conditions météo/luminosité variées (cf. `Applications/pipeline_SH17.ipynb`, extension de l'Axe A, pour la mesure de robustesse en conditions dégradées obtenue en local).

## 2. Impact métier

### Ce que le système change dans les processus existants

- Le contrôle EPI passe d'un contrôle humain ponctuel (visuel, le matin, non exhaustif sur un grand chantier) à une supervision continue sur les zones couvertes par caméra.
- Le rôle du chef de chantier / responsable sécurité évolue : il n'est plus le détecteur de première ligne mais le validateur des alertes remontées par le système, avec un pouvoir de correction (faux positifs signalés, ajustement du seuil de confiance).
- Le système ne remplace pas le contrôle humain sur les EPI non couverts par l'alerte automatique (gants, lunettes) ni sur les usages incorrects d'un EPI porté mais mal ajusté.

### Freins à l'adoption identifiés

- **Acceptabilité par les travailleurs** : un système de vidéosurveillance perçu comme un contrôle individuel peut générer de la défiance. Nécessite une communication claire sur l'usage des données (alerte de sécurité, pas d'évaluation individuelle) et un cadrage RGPD/droit à l'image en amont du déploiement.
- **Fiabilité perçue** : un taux de faux positifs trop élevé (relance excessive) dégrade la confiance dans le système et peut conduire à l'ignorer. Le choix d'un modèle conservateur (précision > rappel) limite ce risque mais ne l'annule pas.
- **Contraintes d'infrastructure chantier** : connectivité réseau souvent limitée sur site, alimentation électrique des caméras/postes de calcul, conditions d'éclairage et de poussière qui dégradent la qualité d'image par rapport aux images d'entraînement SH17.
- **Responsabilité juridique** : une alerte manquée par le système ne doit pas déresponsabiliser le contrôle humain existant (obligation légale de fourniture et de contrôle des EPI par l'employeur, Code du travail) ; le système doit être positionné comme une aide à la décision, pas un substitut de responsabilité.

### Analyse SWOT

| | Positif | Négatif |
|---|---|---|
| **Interne** | **Forces** : modèle temps réel (yolo11l, latence mesurée en Axe A), rappel priorisé sur le casque (EPI le plus critique), pipeline reproductible et documenté (seed fixée, EDA justifiée) | **Faiblesses** : rappel insuffisant sur `safety-vest` pour une alerte ferme, robustesse mesurée en conditions dégradées **insuffisante** (mAP@50-95 -0.133 en absolu, soit environ -30 % en relatif, sur un sous-ensemble faible luminosité + occlusion synthétique — cf. Axe A extension), pas de tracking multi-frame (une alerte par frame, pas de persistance par individu) |
| **Externe** | **Opportunités** : cadre réglementaire favorable (obligation EPI déjà en vigueur, facilite la justification du déploiement), dataset et outils open source réutilisables pour étendre à d'autres EPI ou d'autres sites | **Menaces** : acceptabilité sociale du contrôle vidéo, dépendance à la qualité/position des caméras existantes sur chantier, dérive des conditions réelles par rapport au dataset d'entraînement (SH17 n'est pas exclusivement du BTP) |

## 3. Feuille de route de déploiement

### Phase 1 — Test (0-2 mois)

- Déploiement sur le chantier pilote, une seule zone couverte, alerte casque uniquement (gilet en mode indicatif non affiché à l'opérateur pour ne pas générer de bruit).
- Validation manuelle systématique des alertes par le responsable sécurité pendant les 2 premières semaines (taux de faux positifs/négatifs mesuré en conditions réelles, à comparer aux métriques du jeu de test SH17).
- Indicateur de succès : taux de faux positifs perçu comme acceptable par l'équipe terrain (seuil à définir avec le client, ex. < 1 fausse alerte validée par heure de fonctionnement), et absence de non-détection sur les cas rapportés a posteriori par l'équipe sécurité.

### Phase 2 — Généralisation (2-5 mois)

- Extension à l'ensemble des zones à risque du chantier pilote, puis à un second chantier de profil différent (taille, luminosité, type de travaux).
- Réintégration du gilet comme indice affiché aux opérateurs, avec suivi du taux de rappel réel avant de basculer en alerte ferme.
- Mise en place du tableau de bord de suivi (taux de conformité, alertes/jour, par zone) pour objectiver la tendance dans le temps et informer les formations ciblées.
- Indicateur de succès : couverture de 100 % des zones à risque identifiées, tableau de bord utilisé effectivement par le responsable sécurité (adoption mesurée par la fréquence de consultation).

### Phase 3 — Optimisation (5-9 mois)

- Ré-entraînement du modèle avec les données terrain collectées (conditions réelles de chantier, corrections des faux positifs/négatifs remontés), pour réduire l'écart entre performance sur SH17 et performance en conditions réelles.
- Étude de la piste tracking multi-frame (persistance de l'alerte par individu plutôt que par frame isolée) pour réduire le bruit et fiabiliser la remontée.
- Étude d'une distillation ou d'un modèle plus léger si un déploiement edge (caméra embarquée, sans serveur central) devient un objectif.
- Indicateur de succès : amélioration mesurable du rappel `safety-vest` par rapport à la version initiale, décision argumentée sur l'extension à d'autres EPI (gants, lunettes) ou d'autres sites.

## Limites et périmètre

Cette feuille de route est une proposition de cadrage, pas un engagement contractuel : les seuils chiffrés (faux positifs/heure, durées de phase) sont des points de départ à valider avec les parties prenantes métier (équipe sécurité, direction) avant tout déploiement réel, conformément à la démarche de validation attendue en section 5.3 du cahier des charges.
