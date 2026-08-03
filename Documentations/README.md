# Détection automatique des EPI sur chantier (dataset SH17)

Système de détection du port des Équipements de Protection Individuelle (casque, gilet) à partir d'images ou de flux vidéo, avec déclenchement d'une alerte de non-conformité. Projet de détection d'objets fondé sur YOLO11, entraîné sur le dataset SH17 (17 classes, environ 8000 images).

## Aperçu

Le pipeline complet couvre : réorganisation du dataset (train / val / test), exploration et statistiques, nettoyage (images floues, annotations invalides), pré-traitement justifié, augmentation, choix et entraînement du modèle, optimisation des hyperparamètres, évaluation par classe, et inférence (image avec bandeau d'alerte, vidéo annotée). Deux axes d'analyse avancée sont traités (comparaison de modèles étendue à l'empreinte énergétique et à la robustesse, focus 3 classes EPI), une interface de démonstration Streamlit avec tableau de bord, ainsi qu'une stratégie d'intégration IA ([`Documentations/STRATEGIE_INTEGRATION.md`](Documentations/STRATEGIE_INTEGRATION.md)).

## Structure du dépôt

```
.
├── README.md
├── .gitignore
├── Applications/
│   ├── pipeline_SH17.ipynb       # pipeline reproductible de bout en bout
│   ├── requirements.txt          # dépendances du pipeline complet (notebook)
│   └── app/
│       ├── app.py                # interface Streamlit (Axe E)
│       ├── requirements.txt      # dépendances légères pour la seule démo
│       └── weights/best.pt       # copie du modèle retenu (non versionnée)
├── Modèles/
│   ├── train.py                  # script d'entraînement autonome (modèle final 17 classes)
│   └── weights/best.pt           # copie « sauvegardée » du modèle retenu (non versionnée)
├── Données/
│   └── README.md                 # source du dataset, procédure de nettoyage/split (pas de données versionnées)
└── Documentations/
    ├── STRATEGIE_INTEGRATION.md  # cas d'usage, SWOT, feuille de route de déploiement
    ├── rapport.md / rapport.pdf  # rapport technique structuré (BC02 §6.1)
    ├── BC02 - Projet - 1.pdf     # cahier des charges
    └── Projet_TraitementImage.pdf# ébauche de sujet initiale (historique de cadrage)
```

`README.md` reste à la racine (convention GitHub).
Les poids des modèles, les datasets et les sorties d'entraînement (`SH17_yolo/`, `models/`, `runs/`)
ne sont pas versionnés (voir `.gitignore`) en raison de leur taille et restent à la racine du dépôt.

## Reproduction du pipeline

1. Installer les dépendances :
   ```bash
   pip install -r Applications/requirements.txt
   ```
2. Le dataset SH17 est récupéré automatiquement via `kagglehub` (cellule de configuration du notebook). Aucun téléchargement manuel n'est nécessaire.
3. Ouvrir et exécuter `Applications/pipeline_SH17.ipynb` dans l'ordre (le kernel doit être lancé depuis ce dossier, les chemins du notebook sont relatifs à `Applications/`). Les graines aléatoires sont fixées (`seed = 42`) pour la reproductibilité du split et de l'entraînement.

Note : les cellules d'entraînement et de recherche d'hyperparamètres sont coûteuses (GPU recommandé). Elles peuvent être sautées si les poids sont déjà présents ; les cellules d'évaluation rechargent les modèles depuis le disque. `Modèles/train.py` reproduit l'entraînement du modèle final en ligne de commande (`python Modèles/train.py`, chemins résolus automatiquement quel que soit le répertoire courant).

## Interface de démonstration

L'interface permet de charger une image ou une vidéo, d'afficher les détections et un indicateur global de conformité, et de régler le seuil de confiance. Chaque traitement est journalisé dans `historique.csv` (non versionné) et alimente un onglet **Tableau de bord** : taux de conformité, alertes du jour et tendance, heatmap des non-conformités par zone et par jour (déclarée par l'utilisateur, pas de géolocalisation), timeline des alertes, filtres zone / période / statut. L'application inclut un guide d'utilisation intégré et une bascule de langue FR/EN.

```bash
cd Applications/app
pip install -r requirements.txt
# le best.pt du modèle retenu est déjà copié dans Applications/app/weights/best.pt
streamlit run app.py
```

Le modèle attendu par défaut est `weights/best.pt`, relatif au dossier `Applications/app/` (chemin fixe, non modifiable dans l'interface pour rester simple pour un utilisateur non technique). Déploiement : local uniquement pour l'instant (pas d'hébergement Streamlit Cloud préparé).

## Choix techniques principaux

- **Modèle** : YOLO11 (famille un passage), retenu pour le compromis vitesse / précision et l'écosystème (format d'annotation, tracking natif). Comparé à RT-DETR (transformeur) dans l'Axe A.
- **Modèle retenu pour l'inférence** : yolo11l fine-tuné, choisi pour son rappel supérieur sur les classes EPI rares (contexte sécurité).
- **Résolution d'entrée** : 768 px, motivée par la forte proportion de petits objets observée à l'exploration.
- **Logique de conformité** : une tête détectée non coiffée d'un casque déclenche l'alerte. L'association casque / tête est spatiale (la décision n'est pas une sortie directe du modèle). Le gilet et les gants ne déclenchent pas d'alerte ferme car leur rappel est insuffisant.

## Résultats clés

- **Nettoyage** (section 1.3 du notebook) : détection d'images floues (variance du Laplacien, seuil à recalibrer sur la distribution réelle) et d'annotations invalides (bbox hors image, dimensions nulles, doublons). Code prêt à l'exécution ; nécessite le dataset complet, non versionné ici.
- **Évaluation** (modèle retenu `yolo11l_full-12`, split test) : mAP@50-95 global **0.441** (mAP@50 0.667). La performance par classe est corrélée à la fréquence d'entraînement ; les classes EPI critiques sont rares, d'où une attention particulière au rappel (`helmet` 0.499 et `safety-vest` 0.307 en mAP@50-95, contre `head` 0.730).
- **Axe A** : yolo11l domine RT-DETR sur le mAP tout en étant plus léger et rapide (données historiques, entraînement complet sur serveur H100 aujourd'hui indisponible — cf. note de reproductibilité dans le notebook). yolo11m est un compromis intermédiaire pertinent pour le débit. Extension énergie + robustesse : exécutée réellement en local (GFLOPs, puissance GPU mesurée via `nvidia-smi`, delta de mAP sur images dégradées — faible luminosité, occlusion synthétique).
- **Axe B** : restreindre l'entraînement aux 3 classes de conformité (`helmet`/`head`/`safety-vest`, protocole réduit à 40 époques faute de serveur dédié) dégrade en moyenne leurs performances (mAP@50-95 -0.037), avec un recul net sur `helmet` (-0.087) et `head` (-0.034) ; seul `safety-vest` progresse légèrement (+0.010), au prix d'un rappel en baisse. Résultat contre-intuitif documenté (perte du partage de représentations inter-classes).
- **Optimisation** : la recherche d'hyperparamètres concentre son gain sur les classes EPI rares (rappel du casque amélioré), grâce à un taux d'apprentissage abaissé qui préserve le transfer learning.
- **Stratégie d'intégration** : cas d'usage prioritaires, impact métier, matrice SWOT et feuille de route de déploiement en 3 phases, détaillés dans [`Documentations/STRATEGIE_INTEGRATION.md`](Documentations/STRATEGIE_INTEGRATION.md).

## Données

Dataset SH17 (17 classes : person, head, helmet, safety-vest, etc.). Source : https://github.com/ahmadmughees/SH17dataset

Les trois classes liées à la conformité EPI sont helmet (id 10), head (id 12) et safety-vest (id 16).
