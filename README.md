# Détection automatique des EPI sur chantier (dataset SH17)

Système de détection du port des Équipements de Protection Individuelle (casque, gilet) à partir d'images ou de flux vidéo, avec déclenchement d'une alerte de non-conformité. Projet de détection d'objets fondé sur YOLO11, entraîné sur le dataset SH17 (17 classes, environ 8000 images).

## Aperçu

Le pipeline complet couvre : réorganisation du dataset (train / val / test), exploration et statistiques, pré-traitement justifié, augmentation, choix et entraînement du modèle, optimisation des hyperparamètres, évaluation par classe, et inférence (image avec bandeau d'alerte, vidéo annotée). Deux axes d'analyse avancée sont traités (comparaison de modèles, focus 3 classes EPI), ainsi qu'une interface de démonstration Streamlit.

## Structure du dépôt

```
.
├── README.md
├── requirements.txt              # dépendances du pipeline complet (notebook)
├── .gitignore
├── pipeline_SH17.ipynb           # pipeline reproductible de bout en bout
├── app/
│   ├── app.py                    # interface Streamlit (Axe E)
│   └── requirements.txt          # dépendances légères pour la seule démo
└── rapport.pdf                   # rapport écrit
```

Les poids des modèles, les datasets et les sorties d'entraînement ne sont pas versionnés (voir `.gitignore`) en raison de leur taille.

## Reproduction du pipeline

1. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
2. Le dataset SH17 est récupéré automatiquement via `kagglehub` (cellule de configuration du notebook). Aucun téléchargement manuel n'est nécessaire.
3. Ouvrir et exécuter `pipeline_SH17.ipynb` dans l'ordre. Les graines aléatoires sont fixées (`seed = 42`) pour la reproductibilité du split et de l'entraînement.

Note : les cellules d'entraînement et de recherche d'hyperparamètres sont coûteuses (GPU recommandé). Elles peuvent être sautées si les poids sont déjà présents ; les cellules d'évaluation rechargent les modèles depuis le disque.

## Interface de démonstration

L'interface permet de charger une image ou une vidéo, d'afficher les détections et un indicateur global de conformité, et de régler le seuil de confiance.

```bash
cd app
pip install -r requirements.txt
# placer le best.pt du modèle retenu dans app/weights/best.pt
streamlit run app.py
```

Le modèle attendu par défaut est `app/weights/best.pt` (chemin modifiable dans la barre latérale de l'application).

## Choix techniques principaux

- **Modèle** : YOLO11 (famille un passage), retenu pour le compromis vitesse / précision et l'écosystème (format d'annotation, tracking natif). Comparé à RT-DETR (transformeur) dans l'Axe A.
- **Modèle retenu pour l'inférence** : yolo11l fine-tuné, choisi pour son rappel supérieur sur les classes EPI rares (contexte sécurité).
- **Résolution d'entrée** : 768 px, motivée par la forte proportion de petits objets observée à l'exploration.
- **Logique de conformité** : une tête détectée non coiffée d'un casque déclenche l'alerte. L'association casque / tête est spatiale (la décision n'est pas une sortie directe du modèle). Le gilet et les gants ne déclenchent pas d'alerte ferme car leur rappel est insuffisant.

## Résultats clés

- **Évaluation** : la performance par classe est corrélée à la fréquence d'entraînement ; les classes EPI critiques sont rares, d'où une attention particulière au rappel.
- **Axe A** : yolo11l domine RT-DETR sur le mAP tout en étant plus léger et rapide. yolo11m est un compromis intermédiaire pertinent pour le débit.
- **Axe B** : restreindre l'entraînement aux trois classes de conformité dégrade les performances (perte du partage de représentations inter-classes), résultat contre-intuitif documenté.
- **Optimisation** : la recherche d'hyperparamètres concentre son gain sur les classes EPI rares (rappel du casque amélioré), grâce à un taux d'apprentissage abaissé qui préserve le transfer learning.

## Données

Dataset SH17 (17 classes : person, head, helmet, safety-vest, etc.). Source : https://github.com/ahmadmughees/SH17dataset

Les trois classes liées à la conformité EPI sont helmet (id 10), head (id 12) et safety-vest (id 16).
