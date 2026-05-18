import os
import sys
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from detector import SafetyDetector
from utils import annotate_image, annotate_image_single_model, generate_report

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
VEST_MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")
HELMET_MODEL_PATH = os.path.join(MODELS_DIR, "helmet_best.pt")
FIRE_EXT_MODEL_PATH = os.path.join(MODELS_DIR, "fire_extinguisher_best (2).pt")
GLOVE_MODEL_PATH = os.path.join(MODELS_DIR, "glove_best.pt")

MODEL_CONFIG = {
    "vest": {
        "path": VEST_MODEL_PATH,
        "label": "Safety Vest",
        "arch": "YOLOv8m",
        "default_conf": 0.25,
    },
    "helmet": {
        "path": HELMET_MODEL_PATH,
        "label": "Helmet",
        "arch": "YOLOv8m",
        "default_conf": 0.25,
    },
    "fire_ext": {
        "path": FIRE_EXT_MODEL_PATH,
        "label": "Fire Extinguisher",
        "arch": "YOLO11m",
        "default_conf": 0.35,
    },
    "glove": {
        "path": GLOVE_MODEL_PATH,
        "label": "Glove",
        "arch": "YOLOv8m",
        "default_conf": 0.25,
    },
}

st.set_page_config(
    page_title="Safety Detection",
    page_icon="construction",
    layout="wide",
)


@st.cache_resource
def load_detector():
    available = {}
    missing = []
    for key, cfg in MODEL_CONFIG.items():
        if os.path.exists(cfg["path"]):
            available[key] = cfg["path"]
        else:
            missing.append(f"{cfg['label']} model (`{cfg['path']}`)")

    if missing:
        st.warning(
            "**Missing models (detection disabled for these):**\n\n"
            + "\n\n".join(f"- {m}" for m in missing)
            + "\n\nTrain each model on Kaggle using the notebooks in `notebooks/`, "
            "download the `.pt` files, and place them in the `models/` folder."
        )

    if not available:
        st.error(
            "No models found in `models/` folder. "
            "Add at least one model to run detection."
        )
        st.stop()

    kwargs = {
        f"{key}_model_path": path for key, path in available.items()
        if key in ("vest", "helmet", "fire_ext", "glove")
    }
    return SafetyDetector(**kwargs), available


detector, available_models = load_detector()

# --- Sidebar ---
st.sidebar.header("Model Selection & Thresholds")

enabled_models = {}
model_thresholds = {}
model_iou = {}

for key, cfg in MODEL_CONFIG.items():
    if key not in available_models:
        continue
    with st.sidebar.expander(f"{cfg['label']} ({cfg['arch']})", expanded=True):
        enabled = st.checkbox(
            "Enable", value=True, key=f"enable_{key}",
        )
        conf = st.slider(
            "Confidence Threshold", 0.0, 1.0,
            cfg["default_conf"], 0.05,
            key=f"conf_{key}",
            help=f"Minimum confidence for {cfg['label']} detections",
        )
        iou = st.slider(
            "IOU Threshold", 0.0, 1.0, 0.45, 0.05,
            key=f"iou_{key}",
            help="NMS IOU threshold for overlapping boxes",
        )
        enabled_models[key] = enabled
        model_thresholds[key] = conf
        model_iou[key] = iou

st.sidebar.markdown("---")
avail = detector.available_models
st.sidebar.markdown(f"**Device:** {detector.device}")
for key, cfg in MODEL_CONFIG.items():
    status = cfg["arch"] if key in avail else "Not loaded"
    st.sidebar.markdown(f"**{cfg['label']}:** {status}")

# --- Main ---
st.title("Safety Detection")
st.markdown(
    "Detect safety vest, helmet, fire extinguisher, and glove compliance. "
    "Select models in the sidebar and upload an image below."
)

tab_webcam, tab_upload = st.tabs(["Webcam Capture", "Image Upload"])

MODEL_DISPLAY_ORDER = ["vest", "helmet", "fire_ext", "glove"]

MODEL_EMOJIS = {
    "vest": "Vest",
    "helmet": "Helmet",
    "fire_ext": "FireExt",
    "glove": "Glove",
}

MODEL_EMOJIS = {
    "vest": "🦺",
    "helmet": "⛑️",
    "fire_ext": "🧯",
    "glove": "🧤",
}


def run_detection(image_bgr, image_np):
    active_models = [k for k in MODEL_DISPLAY_ORDER if enabled_models.get(k, False)]

    if not active_models:
        st.warning("No models enabled. Select at least one model in the sidebar.")
        return []

    # --- Run all models first (no output yet) ---
    all_detections = []
    model_results = {}

    for model_key in active_models:
        cfg = MODEL_CONFIG[model_key]
        with st.spinner(f"Running {cfg['label']} model..."):
            conf = model_thresholds[model_key]
            iou = model_iou[model_key]

            if model_key == "vest":
                dets = detector.predict_vest(image_bgr, conf, iou)
            elif model_key == "helmet":
                dets = detector.predict_helmet(image_bgr, conf, iou)
            elif model_key == "fire_ext":
                dets = detector.predict_fire_extinguisher(image_bgr, conf, iou, conf)
            elif model_key == "glove":
                dets = detector.predict_glove(image_bgr, conf, iou)
            else:
                dets = []

        model_results[model_key] = dets
        all_detections.extend(dets)

    # --- Top section: original + combined annotated side by side ---
    combined_bgr = annotate_image(image_bgr, all_detections)
    combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(image_np, caption="Original Image", use_container_width=True)
    with img_col2:
        st.image(combined_rgb, caption="All Models Combined", use_container_width=True)

    # --- Save original picture ---
    SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "testing_all_models")
    if st.button("Save Picture"):
        os.makedirs(SAVE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, f"capture_{timestamp}.jpg")
        cv2.imwrite(filename, image_bgr)
        st.success(f"Picture saved to `{filename}`")

    # --- Summary directly under the images ---
    st.subheader("Combined Summary")
    summary = detector.get_summary(all_detections)

    summary_cols = []
    if "vest" in active_models:
        summary_cols.append(("Safety Vests", summary["Safety Vest"]))
        summary_cols.append(("Vest Violations", summary["No Safety Vest"]))
        summary_cols.append(("Total People", summary["vest_total"]))
    if "helmet" in active_models:
        summary_cols.append(("Helmets Detected", summary["Safety Helmet"]))
        summary_cols.append(("No Helmet", summary["No Helmet"]))
    if "fire_ext" in active_models:
        summary_cols.append(("Fire Extinguishers", summary["Fire Extinguisher"]))
    if "glove" in active_models:
        summary_cols.append(("Gloves Worn", summary["Glove Wearing"]))
        summary_cols.append(("No Gloves", summary["No Gloves"]))
        glove_total = summary["glove_total"]
        glove_compliance = (
            f"{summary['Glove Wearing'] / glove_total * 100:.1f}%"
            if glove_total > 0 else "N/A"
        )
        summary_cols.append(("Glove Compliance", glove_compliance))

    if summary_cols:
        cols = st.columns(min(len(summary_cols), 4))
        for i, (label, value) in enumerate(summary_cols):
            cols[i % 4].metric(label, value)

    st.markdown("---")

    # --- Individual model results below ---
    st.subheader("Individual Model Results")

    for model_key in active_models:
        cfg = MODEL_CONFIG[model_key]
        dets = model_results[model_key]
        annotated_bgr = annotate_image_single_model(image_bgr, dets, model_key)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        st.markdown(f"**{cfg['label']} ({cfg['arch']})**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detections", len(dets))
            if dets:
                classes_found = set(d["class_name"] for d in dets)
                st.write(", ".join(classes_found))
        with col2:
            st.image(annotated_rgb, caption=f"{cfg['label']} Result", use_container_width=True)

        st.markdown("---")

    if all_detections:
        st.markdown(generate_report(all_detections))

    return all_detections


# --- Webcam Tab ---
with tab_webcam:
    st.subheader("Capture from Webcam")
    captured = st.camera_input("Take a photo")

    if captured is not None:
        pil_image = Image.open(captured).convert("RGB")
        image_np = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        run_detection(image_bgr, image_np)

# --- Upload Tab ---
with tab_upload:
    st.subheader("Upload an Image")
    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        pil_image = Image.open(uploaded).convert("RGB")
        image_np = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        detections = run_detection(image_bgr, image_np)

        if detections:
            rows = []
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                model_label = MODEL_CONFIG.get(det["model"], {}).get("label", det["model"])
                rows.append({
                    "Model": model_label,
                    "Class": det["class_name"],
                    "Confidence": f"{det['confidence']:.2%}",
                    "X1": x1, "Y1": y1, "X2": x2, "Y2": y2,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
