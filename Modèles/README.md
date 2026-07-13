# Modèles

## Architectures comparées (Axe A)

yolo11s, yolo11m, yolo11l, rtdetr-l — voir `Applications/pipeline_SH17.ipynb` (Axe A) pour la
comparaison complète (mAP, latence CPU/GPU, GFLOPs, empreinte énergétique, robustesse).

## Modèle retenu

**`yolo11l_full-12`** : yolo11l fine-tuné sur les 17 classes, résolution 768px, ré-entraîné en local
(poste personnel, sans serveur GPU dédié — 12 tentatives jusqu'à convergence stable après la bascule
hors du serveur H100 initial). Retenu pour son rappel supérieur sur les classes EPI rares
(`helmet`, `safety-vest`) plutôt que pour le mAP global.

Le poids sauvegardé de ce modèle est copié dans `weights/best.pt` (non versionné, voir
`.gitignore`) — copie identique à celle utilisée par l'application (`Applications/app/weights/best.pt`).

## Hyperparamètres retenus

```
imgsz=768, epochs=100, batch=4, optimizer=AdamW, lr0=8e-05, cos_lr=True,
patience=20, seed=42
```

`lr0=8e-05` a été identifié via recherche d'hyperparamètres sur le serveur H100 (`model.tune()`,
cf. notebook §2.3) puis réappliqué pour l'entraînement complet local.

## Reproduction

```bash
python train.py
```

`train.py` résout tous les chemins (dataset, poids pré-entraînés, dossier de sortie) relativement à
la racine du dépôt via `Path(__file__)`, donc exécutable depuis n'importe quel répertoire courant.
Entraînement coûteux (GPU recommandé, plusieurs heures sur une carte 8 Go) ; peut être sauté si
`weights/best.pt` est déjà présent.

## Axe B (focus 3 classes)

Comparaison `yolo11l_3cls` / `yolo11m_3cls` (helmet/head/safety-vest uniquement) réalisée avec un
protocole réduit (40 époques au lieu de 100, faute de serveur GPU dédié) — voir le notebook pour le
détail et la conclusion tirée des résultats réels obtenus.
