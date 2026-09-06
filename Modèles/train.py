"""
Entrainement YOLO11l sur SH17, 17 classes.

RACINE_DEPOT
------------
Ce fichier est livre a deux endroits, et un nombre fixe de `.parent` ne peut
pas convenir aux deux :

    dans le depot     SH17-EPI-detection/Modeles/train.py
                      la racine est le dossier au-dessus
    dans la remise    Projet_1_Detection_EPI/Modeles/A_Gautier_Blairon/train.py
                      le dossier au-dessus est `Modeles/`, qui ne porte rien

La racine cherchee est celle qui porte `SH17_yolo/data.yaml`. On la trouve par
ce repere plutot que par un comptage de niveaux, et le repli est explicite :
une erreur nommee, jamais un chemin faux qui echoue plus loin.

Le jeu SH17 (environ 14 Go) et les poids ne sont pas dans l'archive de remise,
pour les raisons donnees par son LISEZ-MOI. Depuis l'archive, il faut donc
pointer une copie du jeu avec `--racine`. Le mode `--dry-run` resout les
chemins, dit ce qui manque, et n'entraine rien : c'est lui qui sert de preuve
que le script est utilisable la ou il est livre.

Usage :
    python train.py --dry-run
    python train.py --racine /chemin/vers/SH17-EPI-detection
    python train.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Repere de racine et sous-chemins attendus, declares une seule fois pour que
# `--dry-run` et l'entrainement disent exactement la meme chose.
REPERE_RACINE = Path("SH17_yolo") / "data.yaml"
POIDS_DEPART = Path("models") / "yolo11l.pt"
SORTIE_RUNS = Path("runs") / "train"

SEED = 42


CONSEIL_RACINE = (
    "Le jeu SH17 n'est pas dans l'archive de remise (environ 14 Go, voir son\n"
    "LISEZ-MOI). Le telecharger, puis relancer avec\n"
    "  python train.py --racine /chemin/vers/le/depot"
)


def _chercher_racine(demandee: Path | None = None) -> Path | None:
    """Racine d'ou se resolvent le jeu de donnees, les poids et les sorties.

    Retourne None plutot que de lever : depuis l'archive de remise, l'absence
    de racine est le cas normal et non une panne, et c'est precisement ce que
    `--dry-run` doit pouvoir rapporter sans echouer.

    `demandee` court-circuite la recherche : c'est ce que passe `--racine`
    quand le jeu de donnees vit ailleurs que l'archive.
    """
    if demandee is not None:
        return demandee.resolve()
    for ancetre in Path(__file__).resolve().parents:
        if (ancetre / REPERE_RACINE).is_file():
            return ancetre
    return None


def construire_arguments() -> argparse.Namespace:
    analyseur = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    analyseur.add_argument(
        "--racine", type=Path, default=None,
        help="racine portant SH17_yolo/, models/ et runs/ (par defaut, deduite)")
    analyseur.add_argument(
        "--dry-run", action="store_true",
        help="resout les chemins, signale ce qui manque, n'entraine pas")
    return analyseur.parse_args()


def inventaire(racine: Path) -> list[tuple[Path, bool]]:
    """Les entrees dont l'entrainement a besoin, et leur presence."""
    attendus = [racine / REPERE_RACINE, racine / POIDS_DEPART]
    return [(chemin, chemin.exists()) for chemin in attendus]


def main() -> int:
    arguments = construire_arguments()
    racine = _chercher_racine(arguments.racine)

    if racine is None:
        print(f"racine       : introuvable depuis {Path(__file__).resolve()}")
        print(f"  repere cherche : {REPERE_RACINE}")
        print(f"\n{CONSEIL_RACINE}")
        # La resolution a fonctionne, elle conclut a une absence de donnees.
        # C'est un diagnostic, pas une panne : `--dry-run` sort donc a zero.
        return 0 if arguments.dry_run else 1

    print(f"racine       : {racine}")
    manquants = []
    for chemin, present in inventaire(racine):
        print(f"  {'ok     ' if present else 'ABSENT '} {chemin}")
        if not present:
            manquants.append(chemin)

    if arguments.dry_run:
        if manquants:
            print(f"\n{len(manquants)} entree(s) absente(s). "
                  "Les chemins se resolvent, les donnees sont a fournir.")
        else:
            print("\nToutes les entrees sont la, l'entrainement peut demarrer.")
        return 0

    if manquants:
        raise SystemExit(
            f"{len(manquants)} entree(s) absente(s), entrainement impossible. "
            "Relancer avec --dry-run pour le detail.")

    # Import tardif : ultralytics coute une a deux secondes et une CUDA, alors
    # que `--dry-run` et `--help` n'en ont aucun besoin. C'est ce qui rend la
    # resolution de chemins verifiable sans installer la pile d'entrainement.
    from ultralytics import YOLO

    modele = YOLO(str(racine / POIDS_DEPART))
    modele.train(
        data=str(racine / REPERE_RACINE),
        imgsz=768,
        epochs=100,
        batch=4,
        cache=False,
        workers=4,
        optimizer="AdamW",
        lr0=8e-05,
        cos_lr=True,
        patience=20,
        seed=SEED,
        project=str(racine / SORTIE_RUNS), name="yolo11l_full",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
