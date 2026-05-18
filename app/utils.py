import cv2
import numpy as np

COLORS = {
    # Vest classes
    ("vest", 0): (255, 0, 0),      # Red for no_safety_vest
    ("vest", 1): (0, 255, 0),      # Green for safety_vest
    # Helmet classes
    ("helmet", 0): (0, 255, 255),  # Cyan for safety_helmet
    # Fire extinguisher classes (all 7 map to orange)
    ("fire_ext", 0): (0, 165, 255),
    ("fire_ext", 1): (0, 165, 255),
    ("fire_ext", 2): (0, 165, 255),
    ("fire_ext", 3): (0, 165, 255),
    ("fire_ext", 4): (0, 165, 255),
    ("fire_ext", 5): (0, 165, 255),
    ("fire_ext", 6): (0, 165, 255),
    # Glove classes
    ("glove", 0): (0, 255, 255),   # Yellow for Glove Wearing
    ("glove", 1): (255, 0, 255),   # Magenta for No Gloves
}

CLASS_NAMES = {
    ("vest", 0): "No Safety Vest",
    ("vest", 1): "Safety Vest",
    ("helmet", 0): "Safety Helmet",
    ("fire_ext", 0): "Fire Extinguisher",
    ("fire_ext", 1): "Fire Extinguisher",
    ("fire_ext", 2): "Fire Extinguisher",
    ("fire_ext", 3): "Fire Extinguisher",
    ("fire_ext", 4): "Fire Extinguisher",
    ("fire_ext", 5): "Fire Extinguisher",
    ("fire_ext", 6): "Fire Extinguisher",
    ("glove", 0): "Glove Wearing",
    ("glove", 1): "No Gloves",
}


def annotate_image_single_model(image: np.ndarray, detections: list, model_key: str) -> np.ndarray:
    """Annotate image with detections from a single model only."""
    filtered = [d for d in detections if d.get("model") == model_key]
    return _draw_annotations(image.copy(), filtered)


def _draw_annotations(annotated: np.ndarray, detections: list) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id = det["class_id"]
        model = det.get("model", "vest")
        conf = det["confidence"]
        key = (model, cls_id)
        color = COLORS.get(key, (255, 255, 255))
        label = f"{CLASS_NAMES.get(key, 'Unknown')} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return annotated


def annotate_image(image: np.ndarray, detections: list) -> np.ndarray:
    annotated = _draw_annotations(image.copy(), detections)

    # Draw violation warnings on the combined view
    vest_dets = [d for d in detections if d.get("model") == "vest"]
    helmet_dets = [d for d in detections if d.get("model") == "helmet"]
    glove_dets = [d for d in detections if d.get("model") == "glove"]

    h, w = annotated.shape[:2]
    warnings = []
    if vest_dets and not helmet_dets:
        warnings.append(("WARNING: No helmets detected!", (0, 0, 255)))
    if vest_dets and not glove_dets:
        warnings.append(("WARNING: No gloves detected!", (255, 0, 255)))

    for idx, (text, color) in enumerate(warnings):
        y_base = h - 15 - idx * 40
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(annotated, (10, y_base - th - 5),
                       (tw + 20, y_base + 5), color, -1)
        cv2.putText(
            annotated, text, (15, y_base),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )

    return annotated


def generate_report(detections: list) -> str:
    if not detections:
        return "No objects detected."

    vest_dets = [d for d in detections if d.get("model") == "vest"]
    helmet_dets = [d for d in detections if d.get("model") == "helmet"]
    fire_ext_dets = [d for d in detections if d.get("model") == "fire_ext"]
    glove_dets = [d for d in detections if d.get("model") == "glove"]

    lines = [f"Total detections: {len(detections)}"]

    if vest_dets:
        vest_violations = sum(1 for d in vest_dets if d["class_id"] == 0)
        vest_compliant = sum(1 for d in vest_dets if d["class_id"] == 1)
        lines.append(f"Safety Vests: {vest_compliant}")
        lines.append(f"Vest Violations: {vest_violations}")

    if helmet_dets:
        lines.append(f"Helmets Detected: {len(helmet_dets)}")
    elif vest_dets:
        lines.append("Helmet Violations: No helmets detected!")

    if fire_ext_dets:
        lines.append(f"Fire Extinguishers Detected: {len(fire_ext_dets)}")

    if glove_dets:
        glove_compliant = sum(1 for d in glove_dets if d["class_id"] == 0)
        glove_violations = sum(1 for d in glove_dets if d["class_id"] == 1)
        lines.append(f"Gloves Worn: {glove_compliant}")
        lines.append(f"No Gloves: {glove_violations}")
    elif vest_dets:
        lines.append("Glove Violations: No gloves detected!")

    return "\n".join(lines)
