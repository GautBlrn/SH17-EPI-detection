"""Contrôles de la règle de conformité EPI de la voie A.

Ce que ces contrôles couvrent, et ce qu'ils ne couvrent pas
----------------------------------------------------------
Ils portent sur la couche de décision écrite à la main, celle qui transforme des
boîtes en verdict : association casque/tête, garde-fou de position, seuil de
recouvrement, indice de gilet manquant, lecture des sorties du modèle. C'est
précisément la partie que le modèle ne fournit pas et que rien d'autre ne
vérifie.

Ils ne mesurent pas la qualité de détection : aucun poids de modèle n'est
chargé, aucune image n'est lue. Les cas de recouvrement sont écrits avec des
coordonnées dont la fraction se calcule de tête, pour que l'attendu soit
vérifiable à la lecture plutôt que copié d'une exécution.

Ils tournent en une fraction de seconde, sans carte graphique et sans réseau.

    cd Applications && python -m pytest -q
"""

import numpy as np
import pytest

from conformite import (
    HEAD_ID,
    HELMET_ID,
    PERSON_ID,
    SEUIL_RECOUVREMENT,
    VEST_ID,
    assess_compliance,
    evaluer_boites,
    helmet_covers_head,
)

# Tête de référence : carré de 100 sur 100, son milieu est donc à y = 50.
TETE = (0, 0, 100, 100)


# --------------------------------------------------------------------------
# L'association casque / tête
# --------------------------------------------------------------------------
def test_casque_superpose_a_la_tete_coiffe():
    """Recouvrement total : la fraction vaut 1,0, très au-dessus du seuil."""
    assert helmet_covers_head(TETE, (0, 0, 100, 100)) is True


def test_casque_a_cote_de_la_tete_ne_coiffe_pas():
    """Intersection vide : aucune fraction, donc aucune association."""
    assert helmet_covers_head(TETE, (200, 0, 300, 40)) is False


def test_fraction_exactement_au_seuil_coiffe():
    """Casque de 10 x 10 = 100, dont 10 x 3 = 30 tombent dans la tête.

    Soit 0,30 exactement, et la comparaison est un `>=` : le seuil est retenu,
    pas exclu. Ce cas est là pour que le jour où le `>=` devient `>`, un
    contrôle le dise.
    """
    assert SEUIL_RECOUVREMENT == 0.3
    assert helmet_covers_head(TETE, (0, -7, 10, 3)) is True


def test_fraction_sous_le_seuil_ne_coiffe_pas():
    """Même casque de 100, mais 10 x 2 = 20 dans la tête, soit 0,20."""
    assert helmet_covers_head(TETE, (0, -8, 10, 2)) is False


def test_le_garde_fou_vertical_rejette_un_casque_trop_bas():
    """Recouvrement parfait, mais le centre du casque est sous le milieu du
    visage : c'est un casque tenu ou posé, pas un casque porté."""
    casque_bas = (0, 60, 100, 100)   # centre à y = 80, sous le milieu à y = 50
    assert helmet_covers_head(TETE, casque_bas) is False


def test_casque_d_aire_nulle_ne_coiffe_pas():
    """Une boîte plate ne divise pas : la fonction répond sans lever."""
    assert helmet_covers_head(TETE, (10, 10, 10, 20)) is False


def test_le_seuil_est_bien_le_parametre_qui_tranche():
    """Le même casque bascule d'un verdict à l'autre selon le seuil seul."""
    casque = (0, -5, 10, 5)   # 10 x 5 = 50 sur 100, soit 0,50
    assert helmet_covers_head(TETE, casque, thresh=0.4) is True
    assert helmet_covers_head(TETE, casque, thresh=0.6) is False


# --------------------------------------------------------------------------
# Le verdict sur une scène complète
# --------------------------------------------------------------------------
def test_scene_vide_est_conforme_et_sans_compteur():
    """Rien de détecté, rien de signalé. Ce n'est pas une preuve d'absence,
    c'est la convention, et elle doit rester explicite."""
    verdict = evaluer_boites([], [])
    assert verdict == {"conforme": True, "tetes_nues": 0,
                       "indice_gilet_manquant": 0, "n_person": 0, "n_vest": 0}


def test_une_tete_sans_casque_declenche_l_alerte():
    verdict = evaluer_boites([TETE], [HEAD_ID])
    assert verdict["conforme"] is False
    assert verdict["tetes_nues"] == 1


def test_deux_tetes_un_seul_casque_laisse_une_tete_nue():
    """Le casque coiffe la première tête, la seconde est à 200 px de là."""
    boites = [TETE, (200, 0, 300, 100), (0, 0, 100, 100)]
    classes = [HEAD_ID, HEAD_ID, HELMET_ID]
    verdict = evaluer_boites(boites, classes)
    assert verdict["tetes_nues"] == 1
    assert verdict["conforme"] is False


def test_les_classes_hors_perimetre_ne_changent_pas_le_verdict():
    """Un outil et des gants détectés ne pèsent sur aucune décision."""
    boites = [TETE, (0, 0, 100, 100), (10, 10, 20, 20), (30, 30, 40, 40)]
    classes = [HEAD_ID, HELMET_ID, 7, 9]   # 7 = tool, 9 = gloves
    assert evaluer_boites(boites, classes)["conforme"] is True


# --------------------------------------------------------------------------
# L'indice de gilet, qui n'est pas une alerte
# --------------------------------------------------------------------------
def test_indice_gilet_compte_les_personnes_sans_gilet():
    boites = [(0, 0, 10, 10)] * 4
    classes = [PERSON_ID, PERSON_ID, PERSON_ID, VEST_ID]
    assert evaluer_boites(boites, classes)["indice_gilet_manquant"] == 2


def test_indice_gilet_ne_devient_jamais_negatif():
    """Plus de gilets que de personnes arrive (un gilet posé, une personne
    manquée). L'indice est un majorant, il se plancher à zéro."""
    boites = [(0, 0, 10, 10)] * 4
    classes = [PERSON_ID, VEST_ID, VEST_ID, VEST_ID]
    assert evaluer_boites(boites, classes)["indice_gilet_manquant"] == 0


def test_un_gilet_manquant_ne_rend_pas_non_conforme():
    """L'alerte ferme ne porte que sur le casque : le rappel mesuré sur le
    gilet ne permet pas d'en faire une alerte. C'est la décision documentée au
    rapport, et elle doit rester vraie dans le code."""
    verdict = evaluer_boites([(0, 0, 10, 10)], [PERSON_ID])
    assert verdict["indice_gilet_manquant"] == 1
    assert verdict["conforme"] is True


# --------------------------------------------------------------------------
# La lecture des sorties du modèle
# --------------------------------------------------------------------------
class _BoitesFactices:
    """Imite l'objet `Results.boxes` d'Ultralytics : des tenseurs qu'on rapatrie
    par `.cpu().numpy()`. Assez pour vérifier l'adaptateur sans charger YOLO."""

    class _Tenseur:
        def __init__(self, tableau):
            self._tableau = tableau

        def cpu(self):
            return self

        def numpy(self):
            return self._tableau

    def __init__(self, boites, classes):
        self.xyxy = self._Tenseur(np.array(boites, dtype=float))
        self.cls = self._Tenseur(np.array(classes, dtype=float))

    def __len__(self):
        return len(self.cls.numpy())


class _ResultatFactice:
    def __init__(self, boites=None, classes=None):
        self.boxes = None if boites is None else _BoitesFactices(boites, classes)


def test_adaptateur_lit_les_sorties_du_modele():
    """Les classes arrivent en flottants côté Ultralytics : la conversion en
    entier doit survivre au passage, sinon aucune comparaison d'identifiant ne
    tombe juste."""
    resultat = _ResultatFactice([TETE, (200, 0, 300, 100)], [HEAD_ID, HEAD_ID])
    verdict = assess_compliance(resultat)
    assert verdict["tetes_nues"] == 2
    assert verdict["conforme"] is False


def test_adaptateur_sans_aucune_detection():
    """Ultralytics renvoie soit `boxes = None`, soit un conteneur vide. Les deux
    chemins mènent au même verdict."""
    attendu = {"conforme": True, "tetes_nues": 0, "indice_gilet_manquant": 0,
               "n_person": 0, "n_vest": 0}
    assert assess_compliance(_ResultatFactice()) == attendu
    assert assess_compliance(_ResultatFactice([], [])) == attendu


@pytest.mark.parametrize("classe", [HEAD_ID, HELMET_ID, PERSON_ID, VEST_ID])
def test_les_identifiants_de_classe_correspondent_au_jeu_sh17(classe):
    """L'ordre des 17 classes de SH17 est fixe et tout est indexé dessus. Une
    permutation silencieuse ferait passer des têtes pour des casques."""
    from conformite import CLASSES

    attendu = {HEAD_ID: "head", HELMET_ID: "helmet",
               PERSON_ID: "person", VEST_ID: "safety-vest"}
    assert CLASSES[classe] == attendu[classe]
