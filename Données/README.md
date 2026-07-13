# Données

Aucune donnée volumineuse n'est versionnée dans ce dossier (dataset ~14 Go). Ce fichier documente
la source et la procédure ; l'exécution réelle se fait dans `Applications/pipeline_SH17.ipynb`.

## Source

Dataset **SH17** (Safe Human 17), récupéré automatiquement via `kagglehub` dans la cellule de
configuration du notebook (aucun téléchargement manuel requis) :
`kagglehub.dataset_download("mugheesahmad/sh17-dataset-for-ppe-detection")`

8 099 images annotées, 75 994 instances, 17 classes (voir `CLAUDE.md` à la racine pour l'ordre
exact des IDs de classe).

## Split train / val / test

SH17 fournit `train_files.txt` / `val_files.txt` mais pas de split test. Le notebook (§1.1) en
crée un en prélevant `TEST_RATIO = 15 %` du train, `SEED = 42` fixée pour la reproductibilité.
Résultat matérialisé dans `SH17_yolo/` à la racine du dépôt (généré, gitignoré).

## Nettoyage et transformation (§1.3-1.5 du notebook)

- **Flou** : détection par variance du Laplacien (seuil à recalibrer sur la distribution réelle),
  filtrage appliqué uniquement au train (le test/val restent représentatifs des conditions réelles).
- **Annotations invalides** : bbox hors image, dimensions nulles, doublons (IoU) supprimés.
- **Redimensionnement / normalisation** : géré nativement par Ultralytics (letterbox 768px,
  normalisation [0,1]), non recodé manuellement (cf. §1.4 du notebook pour la justification
  détaillée de chaque choix de pré-traitement, y compris ceux volontairement écartés).
- **Augmentation** : mosaic natif YOLO + transformations albumentations documentées §1.5.

## Jeux dérivés

- `SH17_yolo_3cls/` : remapping helmet/head/safety-vest pour l'Axe B (généré par le notebook §Axe B).
- `SH17_yolo_degraded/` : sous-ensemble de test dégradé (faible luminosité + occlusion synthétique)
  pour le test de robustesse de l'Axe A (extension).

Aucun de ces dossiers n'est versionné (voir `.gitignore` à la racine).
