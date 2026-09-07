"""conformite.py : la règle de conformité EPI, en un seul exemplaire.

Pourquoi ce module existe
-------------------------
La règle « une tête sans casque déclenche l'alerte » vivait en double, dans
`app/app.py` et dans la cellule d'inférence de `pipeline_SH17.ipynb`. Les deux
copies étaient identiques au commentaire près, et rien ne le garantissait :
corriger le seuil d'un côté laissait l'autre inchangé sans qu'aucun contrôle ne
le voie. Le rapport du projet 1 signalait la duplication en §6.2 en demandant
qu'elle « reste synchronisée », ce qui est une consigne, pas un mécanisme.

Ce module est ce mécanisme. Il ne dépend ni de Streamlit, ni d'OpenCV, ni
d'Ultralytics : il se charge en quelques millisecondes, sans carte graphique et
sans poids de modèle, ce qui le rend testable (`Applications/tests/`).

Deux niveaux d'entrée
---------------------
`evaluer_boites()` travaille sur des tableaux de boîtes et de classes, donc sur
des valeurs qu'un test peut écrire à la main. `assess_compliance()` en est
l'adaptateur : il lit un objet `Results` d'Ultralytics et délègue. C'est cette
séparation qui permet de vérifier la règle sans faire tourner le modèle.
"""

from __future__ import annotations

# Ordre des 17 classes de SH17, fixe et utilisé partout par identifiant.
CLASSES: tuple[str, ...] = (
    "person", "ear", "ear-mufs", "face", "face-guard", "face-mask",
    "foot", "tool", "glasses", "gloves", "helmet", "hands",
    "head", "medical-suit", "shoes", "safety-suit", "safety-vest",
)

PERSON_ID, HELMET_ID, HEAD_ID, VEST_ID = 0, 10, 12, 16

# Affichage restreint aux classes utiles à la conformité (désencombre l'image).
# Liste et non tuple : c'est ce que `model.predict(classes=...)` attend.
EPI_IDS: list[int] = [PERSON_ID, HELMET_ID, HEAD_ID, VEST_ID]

# Fraction du casque qui doit tomber dans la boîte de tête pour que
# l'association soit retenue. Réglé à 0,3 : un casque vu de profil ne recouvre
# jamais la tête entièrement, et un seuil plus haut manquait ces cas.
SEUIL_RECOUVREMENT = 0.3


def helmet_covers_head(head_box, helmet_box, thresh: float = SEUIL_RECOUVREMENT) -> bool:
    """Le casque coiffe-t-il cette tête ?

    Fraction du casque incluse dans la tête, avec un garde-fou de position : le
    centre du casque doit se situer dans la moitié haute de la boîte de tête.
    Sans ce garde-fou, un casque tenu à la main ou posé au sol sous un visage
    suffisait à valider la conformité.

    Les boîtes sont au format ``(x1, y1, x2, y2)``, origine en haut à gauche,
    donc les ordonnées croissent vers le bas.
    """
    hx1, hy1, hx2, hy2 = head_box
    cx1, cy1, cx2, cy2 = helmet_box
    ix1, iy1 = max(hx1, cx1), max(hy1, cy1)
    ix2, iy2 = min(hx2, cx2), min(hy2, cy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    helmet_area = (cx2 - cx1) * (cy2 - cy1)
    if helmet_area == 0:
        return False
    if (cy1 + cy2) / 2 > (hy1 + hy2) / 2:
        return False
    return (inter / helmet_area) >= thresh


def evaluer_boites(boxes, classes) -> dict:
    """Verdict de conformité à partir des boîtes et de leurs classes.

    `boxes` est une séquence de ``(x1, y1, x2, y2)``, `classes` la séquence des
    identifiants de classe correspondants, dans le même ordre.

    Le gilet ne déclenche pas d'alerte ferme : son rappel mesuré est trop bas
    pour cela. Il est rapporté comme un indice, par la différence entre le
    nombre de personnes et le nombre de gilets détectés, qui est un majorant
    grossier et non un comptage.
    """
    classes = [int(c) for c in classes]
    heads = [b for b, c in zip(boxes, classes) if c == HEAD_ID]
    helmets = [b for b, c in zip(boxes, classes) if c == HELMET_ID]
    bare = sum(1 for h in heads if not any(helmet_covers_head(h, hm) for hm in helmets))
    n_person = sum(1 for c in classes if c == PERSON_ID)
    n_vest = sum(1 for c in classes if c == VEST_ID)
    return {
        "conforme": bare == 0,
        "tetes_nues": bare,
        "indice_gilet_manquant": max(0, n_person - n_vest),
        "n_person": n_person,
        "n_vest": n_vest,
    }


def assess_compliance(result) -> dict:
    """Même verdict, à partir d'un objet `Results` d'Ultralytics.

    Adaptateur : il extrait les boîtes et les classes, puis délègue à
    `evaluer_boites`. Une image sans aucune détection est déclarée conforme, ce
    qui est la convention retenue et non un défaut : le système n'a rien vu, il
    ne signale rien. Une absence de détection n'est pas une preuve d'absence.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return evaluer_boites([], [])
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    return evaluer_boites(boxes, classes)
