"""
app.py : Interface de démonstration EPI (Axe E)
================================================
Détection automatique du port des EPI sur image ou vidéo, avec :
  - chargement d'une image ou d'une vidéo par l'utilisateur
  - affichage des détections (bounding boxes + scores)
  - indicateur global de conformité (conforme / non conforme)
  - slider de réglage du seuil de confiance

Interface volontairement simplifiée pour un utilisateur non technique :
seul le seuil de confiance est réglable, les autres paramètres sont figés.

Lancement (en local) :
    pip install -r requirements.txt
    streamlit run app.py

Le modèle (best.pt du yolo11l tuné) doit être placé dans weights/best.pt.
"""

import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# ============================================================
# Paramètres figés (non exposés à l'utilisateur)
# ============================================================
MODEL_PATH = "weights/best.pt"   # modèle retenu (yolo11l tuné)
IMGSZ_IMG = 768                  # résolution pour les images (qualité)
IMGSZ_VID = 768                  # résolution pour la vidéo (plus rapide sur CPU)
MAX_FRAMES = 750                 # plafond démo (~30 s à 25 fps)

CLASSES = [
    "person", "ear", "ear-mufs", "face", "face-guard", "face-mask",
    "foot", "tool", "glasses", "gloves", "helmet", "hands",
    "head", "medical-suit", "shoes", "safety-suit", "safety-vest",
]
PERSON_ID, HELMET_ID, HEAD_ID, VEST_ID = 0, 10, 12, 16
# Affichage restreint aux classes utiles à la conformité (désencombre l'image)
EPI_IDS = [PERSON_ID, HELMET_ID, HEAD_ID, VEST_ID]


# ============================================================
# Logique de conformité (identique au notebook)
# ============================================================
def helmet_covers_head(head_box, helmet_box, thresh=0.3):
    """Le casque coiffe-t-il cette tête ? Fraction du casque incluse dans la
    tête, avec garde-fou : le casque doit être dans la moitié haute du crâne."""
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


def assess_compliance(result):
    """Renvoie un dict : conforme, nb de têtes nues (alerte ferme),
    indice gilet manquant (indicatif)."""
    if result.boxes is None or len(result.boxes) == 0:
        return {"conforme": True, "tetes_nues": 0, "indice_gilet_manquant": 0}
    boxes = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    heads = [b for b, c in zip(boxes, cls) if c == HEAD_ID]
    helmets = [b for b, c in zip(boxes, cls) if c == HELMET_ID]
    bare = sum(1 for h in heads if not any(helmet_covers_head(h, hm) for hm in helmets))
    n_person = int((cls == PERSON_ID).sum())
    n_vest = int((cls == VEST_ID).sum())
    return {"conforme": bare == 0, "tetes_nues": bare,
            "indice_gilet_manquant": max(0, n_person - n_vest)}


def draw_banner(frame_bgr, info):
    """Incruste un bandeau (vert/rouge) en haut de l'image (modifie en place)."""
    h, w = frame_bgr.shape[:2]
    bh = max(40, h // 12)
    if info["conforme"]:
        color, text = (0, 160, 0), "CONFORME"
    else:
        n = info["tetes_nues"]
        mot = "tete" if n <= 1 else "tetes"
        color, text = (0, 0, 255), f"NON CONFORME : {n} {mot} sans casque"
    cv2.rectangle(frame_bgr, (0, 0), (w, bh), color, -1)
    cv2.putText(frame_bgr, text, (15, int(bh * 0.68)), cv2.FONT_HERSHEY_SIMPLEX,
                bh / 45, (255, 255, 255), 2, cv2.LINE_AA)
    return frame_bgr


def detections_table(result):
    """Tableau classe/score des détections, trié par score décroissant."""
    if result.boxes is None or len(result.boxes) == 0:
        return pd.DataFrame(columns=["classe", "score"])
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    df = pd.DataFrame({"classe": [CLASSES[c] for c in cls],
                       "score": [round(float(x), 3) for x in conf]})
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ============================================================
# Chargement du modèle (mis en cache : une seule fois)
# ============================================================
@st.cache_resource(show_spinner="Chargement du modèle...")
def load_model(path):
    return YOLO(path)


# ============================================================
# Traitement vidéo : annote frame par frame, sauvegarde un .mp4
# ============================================================
# Pour rester fluide sur CPU sans saccade :
#   - inférence en 768 + classes filtrées (plus rapide)
#   - sur grosse vidéo, on saute des frames (stride) MAIS on les
#     reduplique à l'écriture, donc la sortie garde son fps d'origine.
# ============================================================
def process_video(model, in_path, out_path, conf, alert_persist=5,
                  max_frames=None, progress=None):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la vidéo.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # stride auto : vidéo courte -> tout traiter ; longue -> en sauter
    stride = 1 if total <= 900 else max(2, total // MAX_FRAMES)

    # writer au fps PLEIN (pas fps/stride) -> sortie à vitesse réelle
    writer = None
    for fourcc in ("avc1", "mp4v", "XVID"):
        wr = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if wr.isOpened():
            writer = wr
            break
        wr.release()
    if writer is None:
        raise RuntimeError("Aucun codec disponible pour l'encodage.")

    streak = 0
    read_i = inferred = 0
    last_annotated = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if read_i % stride == 0:
            # frame analysée
            res = model.predict(frame, imgsz=IMGSZ_VID, conf=conf,
                                classes=EPI_IDS, verbose=False)[0]
            annotated = res.plot(line_width=1, conf=False, font_size=0.4)
            info = assess_compliance(res)
            streak = 0 if info["conforme"] else streak + 1
            draw_banner(annotated, {"conforme": streak < alert_persist,
                                    "tetes_nues": info["tetes_nues"]})
            last_annotated = annotated
            inferred += 1
        else:
            # frame sautée : on réécrit la dernière annotée (fluidité conservée)
            annotated = last_annotated if last_annotated is not None else frame

        writer.write(annotated)
        read_i += 1
        if progress is not None and total:
            progress.progress(min(1.0, read_i / total))
        if max_frames and inferred >= max_frames:
            break

    cap.release()
    writer.release()
    return out_path, read_i, inferred


# ============================================================
# Interface Streamlit
# ============================================================
st.set_page_config(page_title="Détection EPI chantier", layout="wide")
st.title("Détection automatique des EPI sur chantier")
st.caption("Système de contrôle du port des équipements de protection individuelle (casque, gilet).")

# === Réglage (un seul, pour utilisateur non technique) ===
with st.sidebar:
    st.header("Réglage")
    conf = st.slider(
        "Sensibilité de l'alerte (seuil de confiance)",
        0.05, 0.90, 0.20, 0.05,
        help="Plus bas : le système signale davantage (au risque de fausses "
             "alertes). Plus haut : plus prudent.",
    )
    st.caption("Modèle : yolo11l fine-tuné. Affichage limité aux EPI.")

# === Vérification du modèle ===
if not os.path.exists(MODEL_PATH):
    st.error(f"Modèle introuvable : `{MODEL_PATH}`. Placez le best.pt dans weights/.")
    st.stop()
model = load_model(MODEL_PATH)

# === Choix de la source ===
mode = st.radio("Source", ["Image", "Vidéo"], horizontal=True)

# ============================================================
# Mode IMAGE
# ============================================================
if mode == "Image":
    up = st.file_uploader("Charger une image", type=["jpg", "jpeg", "png"])
    if up is not None:
        data = np.frombuffer(up.read(), np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)   # BGR
        res = model.predict(frame, imgsz=IMGSZ_IMG, conf=conf,
                            classes=EPI_IDS, verbose=False)[0]
        annotated = res.plot(line_width=1, conf=False, font_size=0.4)
        info = assess_compliance(res)
        draw_banner(annotated, info)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(annotated[..., ::-1], caption="Détections", use_container_width=True)
        with col2:
            if info["conforme"]:
                st.success("CONFORME")
            else:
                n = info["tetes_nues"]
                st.error(f"NON CONFORME\n\n{n} tête(s) sans casque")
            if info["indice_gilet_manquant"] > 0:
                st.warning(f"Indice (non bloquant) : {info['indice_gilet_manquant']} "
                           "personne(s) sans gilet détecté")
            st.markdown("**Détections**")
            st.dataframe(detections_table(res), use_container_width=True, height=300)

# ============================================================
# Mode VIDÉO
# ============================================================
else:
    up = st.file_uploader("Charger une vidéo", type=["mp4", "avi", "mov", "mkv"])
    if up is not None:
        tin = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tin.write(up.read())
        tin.close()
        out_path = os.path.join(tempfile.gettempdir(), "epi_annotated.mp4")

        if st.button("Lancer le traitement", type="primary"):
            prog = st.progress(0.0)
            with st.spinner("Traitement en cours..."):
                out_path, n_total, n_inf = process_video(
                    model, tin.name, out_path, conf,
                    max_frames=MAX_FRAMES, progress=prog)
            st.success(f"Terminé : {n_total} frames écrites ({n_inf} analysées).")

            try:
                st.video(out_path)
            except Exception:
                st.info("Lecture inline indisponible (codec). Utilisez le bouton ci-dessous.")
            with open(out_path, "rb") as f:
                st.download_button("Télécharger la vidéo annotée", f,
                                   file_name="epi_annotated.mp4", mime="video/mp4")

st.caption("Modèle : yolo11l (fine-tuné). Alerte fondée sur la règle casque/tête. "
           "Affichage restreint aux classes de conformité pour la lisibilité.")