# Luigi, pour information

Écrit le 7 septembre 2026 par Gautier. Ce dépôt est le bras A du projet 1, tu n'y
travailles pas. Un seul commit, et il te concerne quand même parce qu'il touche
un livrable de la remise commune.

## Ce qui a changé

`Modèles/train.py` ne se lançait plus depuis la remise assemblée.
`ROOT = Path(__file__).parent.parent` visait la racine du dépôt quand le fichier
était à `Modèles/train.py` ; descendu d'un niveau sous
`Modèles/A_Gautier_Blairon/` dans l'archive, `ROOT` valait `Modèles/`, et ni
`models/yolo11l.pt` ni `SH17_yolo/data.yaml` n'existaient plus.

C'était l'indicateur C4.2-3, « les codes implémentés sont fonctionnels et sans
erreur », et c'était aggravé par le LISEZ-MOI qui promettait la régénérabilité
des poids par ce script.

La racine est maintenant trouvée par un repère, `SH17_yolo/data.yaml`, et non par
un comptage de niveaux. Le jeu SH17 pesant 14 Go, il n'est pas dans l'archive :
`--racine` permet de pointer une copie, et `--dry-run` résout les chemins puis
dit ce qui manque sans rien entraîner. L'import d'`ultralytics` est devenu tardif
pour que cette vérification ne demande pas la pile complète.

```bash
python "Projet_1_Detection_EPI/Modèles/A_Gautier_Blairon/train.py" --dry-run
```

Trois autres scripts de `Modèles/` avaient le même défaut, un dans
`epi_detection` et deux dans `projet_nlp_app`. Les quatre sortent maintenant avec
le code 0 depuis `REMISE_BC02.zip` décompressé.

## Où lire la suite

Le détail est dans `JOURNAL_CORRECTION_BC02.md`, à la racine du dépôt `livrables-bc02` sur GitHub. Les notes qui te concernent
vraiment sont les `POUR_LUIGI.md` d'`epi_detection`, de `le-limier` et de
`livrables`.
