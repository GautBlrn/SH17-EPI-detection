"""
app.py : Interface de démonstration EPI (Axe E)
================================================
Détection automatique du port des EPI sur image ou vidéo, avec :
  - chargement d'une image ou d'une vidéo par l'utilisateur
  - affichage des détections (bounding boxes + scores)
  - indicateur global de conformité (conforme / non conforme)
  - slider de réglage du seuil de confiance
  - historique persistant (historique.csv) et tableau de bord (KPI,
    répartition par zone, timeline des alertes, filtres)
  - guide d'utilisation intégré et bascule de langue FR/EN

Interface volontairement simplifiée pour un utilisateur non technique :
seuls le seuil de confiance et la zone du chantier sont à renseigner,
les autres paramètres sont figés.

Lancement (en local) :
    pip install -r requirements.txt
    streamlit run app.py

Le modèle (best.pt du yolo11l tuné) doit être placé dans weights/best.pt.
"""

import os
import tempfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from ultralytics import YOLO

# ============================================================
# Paramètres figés (non exposés à l'utilisateur)
# ============================================================
MODEL_PATH = "weights/best.pt"   # modèle retenu (yolo11l tuné)
IMGSZ_IMG = 768                  # résolution pour les images (qualité)
IMGSZ_VID = 768                  # résolution pour la vidéo (plus rapide sur CPU)
MAX_FRAMES = 750                 # plafond démo (~30 s à 25 fps)
HISTORIQUE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historique.csv")
ZONES_PREDEFINIES = [
    "Zone A - Terrassement", "Zone B - Gros oeuvre",
    "Zone C - Second oeuvre", "Zone D - Stockage/Livraison", "Autre",
]

CLASSES = [
    "person", "ear", "ear-mufs", "face", "face-guard", "face-mask",
    "foot", "tool", "glasses", "gloves", "helmet", "hands",
    "head", "medical-suit", "shoes", "safety-suit", "safety-vest",
]
PERSON_ID, HELMET_ID, HEAD_ID, VEST_ID = 0, 10, 12, 16
# Affichage restreint aux classes utiles à la conformité (désencombre l'image)
EPI_IDS = [PERSON_ID, HELMET_ID, HEAD_ID, VEST_ID]


# ============================================================
# Textes d'interface (FR/EN) — traduction limitée aux libellés
# utilisateur visibles (titres, boutons, statuts) ; la logique
# métier et les commentaires de code restent en français.
# ============================================================
STRINGS = {
    "fr": {
        "title": "Détection automatique des EPI sur chantier",
        "caption": "Système de contrôle du port des équipements de protection individuelle (casque, gilet).",
        "lang_label": "Langue / Language",
        "sidebar_header": "Réglage",
        "slider_label": "Sensibilité de l'alerte (seuil de confiance)",
        "slider_help": ("Plus bas : le système signale davantage (au risque de fausses "
                         "alertes). Plus haut : plus prudent."),
        "model_caption": "Modèle : yolo11l fine-tuné. Affichage limité aux EPI.",
        "guide_title": "Guide d'utilisation",
        "guide_text": (
            "**Image / Vidéo** : choisissez une zone du chantier, chargez un fichier, "
            "puis (pour la vidéo) lancez le traitement. Le seuil de confiance dans la "
            "barre latérale règle la sensibilité : plus il est bas, plus le système "
            "signale de détections (au risque de fausses alertes).\n\n"
            "**Bandeau de conformité** : un bandeau vert *CONFORME* ou rouge "
            "*NON CONFORME* est incrusté en haut de l'image/vidéo dès qu'une tête est "
            "détectée sans casque à proximité. Un indice (non bloquant) de gilet "
            "manquant peut aussi s'afficher.\n\n"
            "**Zone** : la zone est une information déclarative saisie par "
            "l'utilisateur avant traitement — ce n'est pas une géolocalisation "
            "automatique.\n\n"
            "**Tableau de bord** : regroupe l'historique des traitements effectués "
            "dans cette installation (fichier `historique.csv`). Filtrez par zone, "
            "période ou statut pour explorer les tendances de non-conformité."
        ),
        "zone_label": "Zone du chantier",
        "zone_custom_label": "Préciser la zone",
        "mode_label": "Source",
        "mode_image": "Image",
        "mode_video": "Vidéo",
        "mode_dashboard": "Tableau de bord",
        "upload_image": "Charger une image",
        "upload_video": "Charger une vidéo",
        "compliant": "CONFORME",
        "non_compliant": "NON CONFORME",
        "vest_warning_prefix": "Indice (non bloquant)",
        "vest_warning_suffix": "personne(s) sans gilet détecté",
        "detections_label": "**Détections**",
        "run_button": "Lancer le traitement",
        "processing": "Traitement en cours...",
        "download_button": "Télécharger la vidéo annotée",
        "footer_caption": ("Modèle : yolo11l (fine-tuné). Alerte fondée sur la règle casque/tête. "
                            "Affichage restreint aux classes de conformité pour la lisibilité."),
        "model_missing": "Modèle introuvable : `{path}`. Placez le best.pt dans weights/.",
        "dashboard_title": "Tableau de bord de conformité",
        "dashboard_empty": ("Aucune donnée pour l'instant. Traitez une image ou une vidéo "
                             "pour commencer à peupler le tableau de bord."),
        "dashboard_empty_filtered": "Aucune donnée ne correspond à ces filtres.",
        "dashboard_raw_data": "Historique brut",
        "filter_zone": "Zone",
        "filter_period": "Période",
        "filter_alert_type": "Statut",
        "filter_all": "Tous",
        "kpi_compliance_rate": "Taux de conformité",
        "kpi_total_alerts": "Alertes (non conformes)",
        "kpi_alerts_today": "Alertes aujourd'hui",
        "chart_heatmap_title": "Taux de non-conformité par zone",
        "chart_timeline_title": "Alertes par jour",
        "clear_history_button": "Vider l'historique",
        "clear_history_confirm": "Historique supprimé.",
    },
    "en": {
        "title": "Automatic PPE detection on construction sites",
        "caption": "Personal protective equipment compliance monitoring (helmet, vest).",
        "lang_label": "Langue / Language",
        "sidebar_header": "Settings",
        "slider_label": "Alert sensitivity (confidence threshold)",
        "slider_help": ("Lower: the system flags more detections (risk of false "
                         "alerts). Higher: more conservative."),
        "model_caption": "Model: fine-tuned yolo11l. Display limited to PPE classes.",
        "guide_title": "User guide",
        "guide_text": (
            "**Image / Video**: pick a site zone, upload a file, then (for video) "
            "run the processing. The confidence threshold in the sidebar controls "
            "sensitivity: lower means more detections flagged (risk of false "
            "alerts).\n\n"
            "**Compliance banner**: a green *COMPLIANT* or red *NON COMPLIANT* "
            "banner is overlaid on the image/video whenever a head is detected "
            "without a nearby helmet. A non-blocking missing-vest indicator may "
            "also be shown.\n\n"
            "**Zone**: the zone is a declarative field entered by the user before "
            "processing — not automatic geolocation.\n\n"
            "**Dashboard**: aggregates the processing history for this "
            "installation (`historique.csv`). Filter by zone, period or status to "
            "explore non-compliance trends."
        ),
        "zone_label": "Site zone",
        "zone_custom_label": "Specify the zone",
        "mode_label": "Source",
        "mode_image": "Image",
        "mode_video": "Video",
        "mode_dashboard": "Dashboard",
        "upload_image": "Upload an image",
        "upload_video": "Upload a video",
        "compliant": "COMPLIANT",
        "non_compliant": "NON COMPLIANT",
        "vest_warning_prefix": "Indicator (non-blocking)",
        "vest_warning_suffix": "worker(s) without a detected vest",
        "detections_label": "**Detections**",
        "run_button": "Run processing",
        "processing": "Processing...",
        "download_button": "Download annotated video",
        "footer_caption": ("Model: yolo11l (fine-tuned). Alert based on the helmet/head rule. "
                            "Display restricted to compliance classes for readability."),
        "model_missing": "Model not found: `{path}`. Place best.pt in weights/.",
        "dashboard_title": "Compliance dashboard",
        "dashboard_empty": ("No data yet. Process an image or a video to start "
                             "populating the dashboard."),
        "dashboard_empty_filtered": "No data matches these filters.",
        "dashboard_raw_data": "Raw history",
        "filter_zone": "Zone",
        "filter_period": "Period",
        "filter_alert_type": "Status",
        "filter_all": "All",
        "kpi_compliance_rate": "Compliance rate",
        "kpi_total_alerts": "Alerts (non compliant)",
        "kpi_alerts_today": "Alerts today",
        "chart_heatmap_title": "Non-compliance rate by zone",
        "chart_timeline_title": "Alerts per day",
        "clear_history_button": "Clear history",
        "clear_history_confirm": "History cleared.",
    },
}


def t(key):
    """Raccourci de traduction : pioche le libellé dans la langue active."""
    return STRINGS[st.session_state.get("lang", "fr")][key]


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
    indice gilet manquant (indicatif), et les comptages bruts personne/gilet
    (utiles pour l'historique et le tableau de bord)."""
    if result.boxes is None or len(result.boxes) == 0:
        return {"conforme": True, "tetes_nues": 0, "indice_gilet_manquant": 0,
                "n_person": 0, "n_vest": 0}
    boxes = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    heads = [b for b, c in zip(boxes, cls) if c == HEAD_ID]
    helmets = [b for b, c in zip(boxes, cls) if c == HELMET_ID]
    bare = sum(1 for h in heads if not any(helmet_covers_head(h, hm) for hm in helmets))
    n_person = int((cls == PERSON_ID).sum())
    n_vest = int((cls == VEST_ID).sum())
    return {"conforme": bare == 0, "tetes_nues": bare,
            "indice_gilet_manquant": max(0, n_person - n_vest),
            "n_person": n_person, "n_vest": n_vest}


def draw_banner(frame_bgr, info, lang="fr"):
    """Incruste un bandeau (vert/rouge) en haut de l'image (modifie en place).

    Choix accessibilité : le rouge/vert est conservé (code couleur du danger
    universellement reconnu) mais n'est jamais le seul signal — le texte
    explicite ("CONFORME" / "NON CONFORME ...") porte l'information, ce qui
    reste lisible pour les daltoniens (recommandation WCAG : ne pas s'appuyer
    sur la couleur seule)."""
    h, w = frame_bgr.shape[:2]
    bh = max(40, h // 12)
    if info["conforme"]:
        color, text = (0, 160, 0), STRINGS[lang]["compliant"]
    else:
        n = info["tetes_nues"]
        if lang == "en":
            text = f"{STRINGS[lang]['non_compliant']}: {n} bare head{'s' if n != 1 else ''}"
        else:
            mot = "tete" if n <= 1 else "tetes"
            text = f"{STRINGS[lang]['non_compliant']} : {n} {mot} sans casque"
        color = (0, 0, 255)
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
# Historique persistant (pour le tableau de bord)
# ============================================================
def log_event(zone, mode, info):
    """Ajoute une ligne à l'historique persistant (app/historique.csv)."""
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "zone": zone,
        "mode": mode,
        "conforme": bool(info["conforme"]),
        "tetes_nues": int(info["tetes_nues"]),
        "n_person": int(info.get("n_person", 0)),
        "n_vest": int(info.get("n_vest", 0)),
        "indice_gilet_manquant": int(info["indice_gilet_manquant"]),
    }
    file_exists = os.path.exists(HISTORIQUE_PATH)
    pd.DataFrame([row]).to_csv(HISTORIQUE_PATH, mode="a", header=not file_exists, index=False)


def zone_picker(key):
    """Sélecteur de zone (déclaratif, pas de géolocalisation)."""
    choice = st.selectbox(t("zone_label"), ZONES_PREDEFINIES, key=f"zone_select_{key}")
    if choice == "Autre":
        custom = st.text_input(t("zone_custom_label"), key=f"zone_custom_{key}")
        return custom.strip() or "Non renseignée"
    return choice


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
                  max_frames=None, progress=None, lang="fr"):
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
    # Agrégats pour l'historique (le pire état rencontré sur la vidéo)
    agg_bare_max = agg_n_person_max = agg_n_vest_max = 0
    agg_non_conforme_frames = 0
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
            agg_bare_max = max(agg_bare_max, info["tetes_nues"])
            agg_n_person_max = max(agg_n_person_max, info["n_person"])
            agg_n_vest_max = max(agg_n_vest_max, info["n_vest"])
            if not info["conforme"]:
                agg_non_conforme_frames += 1
            streak = 0 if info["conforme"] else streak + 1
            draw_banner(annotated, {"conforme": streak < alert_persist,
                                    "tetes_nues": info["tetes_nues"]}, lang=lang)
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
    agg_info = {
        "conforme": agg_non_conforme_frames == 0,
        "tetes_nues": agg_bare_max,
        "n_person": agg_n_person_max,
        "n_vest": agg_n_vest_max,
        "indice_gilet_manquant": max(0, agg_n_person_max - agg_n_vest_max),
    }
    return out_path, read_i, inferred, agg_info


# ============================================================
# Tableau de bord : KPI, répartition par zone, timeline, filtres
# ============================================================
def render_dashboard():
    st.subheader(t("dashboard_title"))
    if not os.path.exists(HISTORIQUE_PATH):
        st.info(t("dashboard_empty"))
        return
    hist = pd.read_csv(HISTORIQUE_PATH)
    if hist.empty:
        st.info(t("dashboard_empty"))
        return
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    hist["date"] = hist["timestamp"].dt.date
    hist["conforme"] = hist["conforme"].astype(bool)

    # === Filtres ===
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        zones = sorted(hist["zone"].dropna().unique().tolist())
        sel_zones = st.multiselect(t("filter_zone"), zones, default=zones)
    with col_f2:
        min_d, max_d = hist["date"].min(), hist["date"].max()
        date_range = st.date_input(t("filter_period"), value=(min_d, max_d),
                                    min_value=min_d, max_value=max_d)
    with col_f3:
        alert_choice = st.selectbox(
            t("filter_alert_type"),
            [t("filter_all"), t("non_compliant"), t("compliant")],
        )

    filtered = hist[hist["zone"].isin(sel_zones)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        d0, d1 = date_range
        filtered = filtered[(filtered["date"] >= d0) & (filtered["date"] <= d1)]
    if alert_choice == t("non_compliant"):
        filtered = filtered[~filtered["conforme"]]
    elif alert_choice == t("compliant"):
        filtered = filtered[filtered["conforme"]]

    if filtered.empty:
        st.warning(t("dashboard_empty_filtered"))
        return

    # === KPIs ===
    taux_conformite = 100 * filtered["conforme"].mean()
    total_alertes = int((~filtered["conforme"]).sum())
    today = pd.Timestamp.now().date()
    alertes_jour = int(((~filtered["conforme"]) & (filtered["date"] == today)).sum())

    # Tendance : taux de conformité de la période filtrée vs période équivalente précédente
    span_days = (filtered["date"].max() - filtered["date"].min()).days + 1
    prev_end = filtered["date"].min() - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=span_days - 1)
    prev_period = hist[(hist["date"] >= prev_start) & (hist["date"] <= prev_end)]
    delta = None
    if len(prev_period) > 0:
        delta = taux_conformite - 100 * prev_period["conforme"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric(t("kpi_compliance_rate"), f"{taux_conformite:.0f}%",
              delta=f"{delta:+.0f} pts" if delta is not None else None)
    c2.metric(t("kpi_total_alerts"), total_alertes)
    c3.metric(t("kpi_alerts_today"), alertes_jour)

    # === Heatmap des zones à risque (zone x jour de semaine, taux de non-conformité) ===
    # Palette accessible daltoniens : Viridis (perceptuellement uniforme, pas de rouge/vert seul).
    days = (["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
            if lang == "fr" else
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    heat_data = filtered.copy()
    heat_data["jour"] = pd.Categorical(
        heat_data["timestamp"].dt.dayofweek.map(dict(enumerate(days))),
        categories=days, ordered=True)
    heat = (heat_data.groupby(["zone", "jour"], observed=False)["conforme"]
            .apply(lambda s: 100 * (1 - s.mean()) if len(s) else np.nan)
            .reset_index(name="taux_non_conformite"))
    pivot = heat.pivot(index="zone", columns="jour", values="taux_non_conformite")
    fig_zone = px.imshow(pivot, color_continuous_scale="Viridis", aspect="auto",
                          labels={"color": "%"}, title=t("chart_heatmap_title"))
    st.plotly_chart(fig_zone, use_container_width=True)

    # === Timeline des alertes ===
    par_jour = (filtered[~filtered["conforme"]].groupby("date").size()
                .reset_index(name="alertes"))
    fig_time = px.line(par_jour, x="date", y="alertes", markers=True,
                        title=t("chart_timeline_title"),
                        color_discrete_sequence=["#1b6ca8"])
    st.plotly_chart(fig_time, use_container_width=True)

    with st.expander(t("dashboard_raw_data")):
        st.dataframe(filtered.sort_values("timestamp", ascending=False),
                     use_container_width=True)

    if st.button(t("clear_history_button")):
        os.remove(HISTORIQUE_PATH)
        st.success(t("clear_history_confirm"))
        st.rerun()


# ============================================================
# Interface Streamlit
# ============================================================
st.set_page_config(page_title="Détection EPI chantier", layout="wide")

# === Réglages (langue + seuil) — déterminés avant le reste de la page ===
with st.sidebar:
    st.session_state["lang"] = "en" if st.radio(
        STRINGS["fr"]["lang_label"], ["fr", "en"], horizontal=True,
        format_func=lambda k: "Français" if k == "fr" else "English",
    ) == "en" else "fr"
    lang = st.session_state["lang"]
    st.header(t("sidebar_header"))
    conf = st.slider(t("slider_label"), 0.05, 0.90, 0.20, 0.05, help=t("slider_help"))
    st.caption(t("model_caption"))

st.title(t("title"))
st.caption(t("caption"))
with st.expander(t("guide_title"), expanded=False):
    st.markdown(t("guide_text"))

# === Choix de la source (Image / Vidéo / Tableau de bord) ===
mode_keys = ["image", "video", "dashboard"]
mode = st.radio(t("mode_label"), mode_keys,
                format_func=lambda k: t(f"mode_{k}"), horizontal=True)

# === Vérification du modèle (uniquement nécessaire hors tableau de bord) ===
if mode in ("image", "video"):
    if not os.path.exists(MODEL_PATH):
        st.error(t("model_missing").format(path=MODEL_PATH))
        st.stop()
    model = load_model(MODEL_PATH)

# ============================================================
# Mode IMAGE
# ============================================================
if mode == "image":
    zone = zone_picker("image")
    up = st.file_uploader(t("upload_image"), type=["jpg", "jpeg", "png"])
    if up is not None:
        data = np.frombuffer(up.read(), np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)   # BGR
        res = model.predict(frame, imgsz=IMGSZ_IMG, conf=conf,
                            classes=EPI_IDS, verbose=False)[0]
        annotated = res.plot(line_width=1, conf=False, font_size=0.4)
        info = assess_compliance(res)
        draw_banner(annotated, info, lang=lang)
        log_event(zone, "image", info)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(annotated[..., ::-1], caption=t("detections_label"), use_container_width=True)
        with col2:
            if info["conforme"]:
                st.success(t("compliant"))
            else:
                n = info["tetes_nues"]
                st.error(f"{t('non_compliant')}\n\n{n} tête(s) sans casque")
            if info["indice_gilet_manquant"] > 0:
                st.warning(f"{t('vest_warning_prefix')} : {info['indice_gilet_manquant']} "
                           f"{t('vest_warning_suffix')}")
            st.markdown(t("detections_label"))
            st.dataframe(detections_table(res), use_container_width=True, height=300)

# ============================================================
# Mode VIDÉO
# ============================================================
elif mode == "video":
    zone = zone_picker("video")
    up = st.file_uploader(t("upload_video"), type=["mp4", "avi", "mov", "mkv"])
    if up is not None:
        tin = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tin.write(up.read())
        tin.close()
        out_path = os.path.join(tempfile.gettempdir(), "epi_annotated.mp4")

        if st.button(t("run_button"), type="primary"):
            prog = st.progress(0.0)
            with st.spinner(t("processing")):
                out_path, n_total, n_inf, agg_info = process_video(
                    model, tin.name, out_path, conf,
                    max_frames=MAX_FRAMES, progress=prog, lang=lang)
            log_event(zone, "vidéo", agg_info)
            st.success(f"{n_total} frames écrites ({n_inf} analysées).")

            try:
                st.video(out_path)
            except Exception:
                st.info("Lecture inline indisponible (codec). Utilisez le bouton ci-dessous.")
            with open(out_path, "rb") as f:
                st.download_button(t("download_button"), f,
                                   file_name="epi_annotated.mp4", mime="video/mp4")

# ============================================================
# Mode TABLEAU DE BORD
# ============================================================
else:
    render_dashboard()

st.caption(t("footer_caption"))
