import os
import sys

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from detector import SafetyDetector
from utils import annotate_image, generate_report

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
VEST_MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")
HELMET_MODEL_PATH = os.path.join(MODELS_DIR, "helmet_best.pt")
FIRE_EXT_MODEL_PATH = os.path.join(MODELS_DIR, "fire_extinguisher_best.pt")
GLOVE_MODEL_PATH = os.path.join(MODELS_DIR, "glove_best.pt")

st.set_page_config(
    page_title="Safety Detection",
    page_icon="construction",
    layout="wide",
)


@st.cache_resource
def load_detector():
    missing = []
    if not os.path.exists(VEST_MODEL_PATH):
        missing.append(f"Vest model (`models/best.pt`)")
    if not os.path.exists(HELMET_MODEL_PATH):
        missing.append(f"Helmet model (`models/helmet_best.pt`)")
    if not os.path.exists(FIRE_EXT_MODEL_PATH):
        missing.append(f"Fire extinguisher model (`models/fire_extinguisher_best.pt`)")
    if not os.path.exists(GLOVE_MODEL_PATH):
        missing.append(f"Glove model (`models/glove_best.pt`)")

    if missing:
        st.error(
            "Missing models:\n\n"
            + "\n\n".join(f"- {m}" for m in missing)
            + "\n\nTrain each model on Kaggle using the notebooks in `notebooks/`, "
            "download the `.pt` files, and place them in the `models/` folder."
        )
        st.stop()

    return SafetyDetector(VEST_MODEL_PATH, HELMET_MODEL_PATH, FIRE_EXT_MODEL_PATH, GLOVE_MODEL_PATH)


detector = load_detector()

# --- Sidebar ---
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, 0.25, 0.05,
    help="Minimum confidence score for detections",
)
iou_threshold = st.sidebar.slider(
    "IOU Threshold", 0.0, 1.0, 0.45, 0.05,
    help="NMS IOU threshold for overlapping boxes",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Device:** {detector.device}\n\n"
    "**Vest Model:** YOLOv8m\n\n"
    "**Helmet Model:** YOLOv8m\n\n"
    "**Fire Ext. Model:** YOLOv8s\n\n"
    "**Glove Model:** YOLOv8m\n\n"
    "**Classes:** No Safety Vest, Safety Vest, Safety Helmet, Fire Extinguisher, Glove Wearing, No Gloves"
)

# --- Main ---
st.title("Safety Detection")
st.markdown("Detect safety vest, helmet, fire extinguisher, and glove compliance using webcam capture or image upload.")

tab_webcam, tab_upload = st.tabs(["Webcam Capture", "Image Upload"])


def run_detection(image_bgr, image_np):
    with st.spinner("Detecting..."):
        detections = detector.detect_all(
            image_bgr,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

    annotated_bgr = annotate_image(image_bgr, detections)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image_np, caption="Original", use_container_width=True)
    with col2:
        st.image(annotated_rgb, caption="Detected", use_container_width=True)

    st.markdown("---")
    summary = detector.get_summary(detections)

    c1, c2, c3 = st.columns(3)
    c1.metric("Safety Vests", summary["Safety Vest"])
    c2.metric("Vest Violations", summary["No Safety Vest"])
    c3.metric("Total People", summary["vest_total"])

    c4, c5, c6 = st.columns(3)
    c4.metric("Helmets Detected", summary["Safety Helmet"])
    c5.metric("No Helmet", summary["No Helmet"])
    c6.metric("Fire Extinguishers", summary["Fire Extinguisher"])

    c7, c8, c9 = st.columns(3)
    c7.metric("Gloves Worn", summary["Glove Wearing"])
    c8.metric("No Gloves", summary["No Gloves"])
    glove_compliance = (
        f"{summary['Glove Wearing'] / summary['glove_total'] * 100:.1f}%"
        if summary["glove_total"] > 0 else "N/A"
    )
    c9.metric("Glove Compliance", glove_compliance)

    if detections:
        st.markdown(generate_report(detections))

    return detections


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
                rows.append({
                    "Category": det["model"].capitalize(),
                    "Class": det["class_name"],
                    "Confidence": f"{det['confidence']:.2%}",
                    "X1": x1, "Y1": y1, "X2": x2, "Y2": y2,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
