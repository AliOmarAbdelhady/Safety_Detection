import torch
import numpy as np
from ultralytics import YOLO


class SafetyDetector:
    VEST_CLASSES = {0: "No Safety Vest", 1: "Safety Vest"}
    VEST_COLORS = {
        0: (255, 0, 0),   # Red for violations
        1: (0, 255, 0),   # Green for compliant
    }
    HELMET_CLASSES = {0: "Safety Helmet"}
    HELMET_COLORS = {
        0: (0, 255, 255),  # Cyan for helmet detected
    }
    FIRE_EXT_CLASSES = {
        0: "Fire Extinguisher", 1: "Fire Extinguisher", 2: "Fire Extinguisher",
        3: "Fire Extinguisher", 4: "Fire Extinguisher", 5: "Fire Extinguisher",
        6: "Fire Extinguisher",
    }
    GLOVE_CLASSES = {0: "Glove Wearing", 1: "No Gloves"}

    # Fire extinguisher post-processing
    FIRE_EXT_MIN_CONF = 0.35  # Minimum confidence after retrain with YOLO11m

    def __init__(self, vest_model_path=None, helmet_model_path=None,
                 fire_ext_model_path=None, glove_model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.available_models = set()

        if vest_model_path:
            self.vest_model = YOLO(vest_model_path)
            self.vest_model.to(self.device)
            self.available_models.add("vest")
        if helmet_model_path:
            self.helmet_model = YOLO(helmet_model_path)
            self.helmet_model.to(self.device)
            self.available_models.add("helmet")
        if fire_ext_model_path:
            self.fire_ext_model = YOLO(fire_ext_model_path)
            self.fire_ext_model.to(self.device)
            self.available_models.add("fire_ext")
        if glove_model_path:
            self.glove_model = YOLO(glove_model_path)
            self.glove_model.to(self.device)
            self.available_models.add("glove")

    def _run_model(self, model, image, conf_threshold, iou_threshold):
        results = model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False,
        )
        return results[0]

    def _parse_detections(self, result, class_map, model_name):
        detections = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        boxes = result.boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            detections.append({
                "class_id": cls_id,
                "class_name": class_map.get(cls_id, f"class_{cls_id}"),
                "confidence": conf,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "model": model_name,
            })
        return detections

    def predict_vest(self, image, conf_threshold=0.25, iou_threshold=0.45):
        result = self._run_model(self.vest_model, image, conf_threshold, iou_threshold)
        return self._parse_detections(result, self.VEST_CLASSES, "vest")

    def predict_helmet(self, image, conf_threshold=0.25, iou_threshold=0.45):
        result = self._run_model(self.helmet_model, image, conf_threshold, iou_threshold)
        return self._parse_detections(result, self.HELMET_CLASSES, "helmet")

    def _filter_fire_ext_detections(self, detections, min_conf):
        """Post-process fire extinguisher detections to remove false positives."""
        filtered = []
        for det in detections:
            if det["confidence"] >= min_conf:
                filtered.append(det)
        return filtered

    def predict_fire_extinguisher(self, image, conf_threshold=0.25, iou_threshold=0.45,
                                  fire_ext_min_conf=0.35):
        result = self._run_model(self.fire_ext_model, image, conf_threshold, iou_threshold)
        detections = self._parse_detections(result, self.FIRE_EXT_CLASSES, "fire_ext")
        return self._filter_fire_ext_detections(detections, fire_ext_min_conf)

    def predict_glove(self, image, conf_threshold=0.25, iou_threshold=0.45):
        result = self._run_model(self.glove_model, image, conf_threshold, iou_threshold)
        return self._parse_detections(result, self.GLOVE_CLASSES, "glove")

    def detect_all(self, image, conf_threshold=0.25, iou_threshold=0.45,
                   fire_ext_min_conf=0.35):
        all_detections = []
        if "vest" in self.available_models:
            all_detections += self.predict_vest(image, conf_threshold, iou_threshold)
        if "helmet" in self.available_models:
            all_detections += self.predict_helmet(image, conf_threshold, iou_threshold)
        if "fire_ext" in self.available_models:
            all_detections += self.predict_fire_extinguisher(
                image, conf_threshold, iou_threshold, fire_ext_min_conf)
        if "glove" in self.available_models:
            all_detections += self.predict_glove(image, conf_threshold, iou_threshold)
        return all_detections

    def get_summary(self, detections):
        summary = {
            "Safety Vest": 0,
            "No Safety Vest": 0,
            "Safety Helmet": 0,
            "No Helmet": 0,
            "Fire Extinguisher": 0,
            "Glove Wearing": 0,
            "No Gloves": 0,
            "total": 0,
            "vest_total": 0,
            "helmet_total": 0,
            "fire_ext_total": 0,
            "glove_total": 0,
        }
        for det in detections:
            name = det["class_name"]
            if name in summary:
                summary[name] += 1
            summary["total"] += 1
            if det["model"] == "vest":
                summary["vest_total"] += 1
            elif det["model"] == "helmet":
                summary["helmet_total"] += 1
            elif det["model"] == "fire_ext":
                summary["fire_ext_total"] += 1
            elif det["model"] == "glove":
                summary["glove_total"] += 1

        # Flag no-helmet if helmets detected are 0 but vest detections exist (people present)
        if "vest" in self.available_models and "helmet" in self.available_models:
            vest_dets = [d for d in detections if d["model"] == "vest"]
            helmet_dets = [d for d in detections if d["model"] == "helmet"]
            if vest_dets and not helmet_dets:
                summary["No Helmet"] = summary["vest_total"]

        # Flag no-gloves if glove detections are 0 but vest detections exist (people present)
        if "vest" in self.available_models and "glove" in self.available_models:
            vest_dets = [d for d in detections if d["model"] == "vest"]
            glove_dets = [d for d in detections if d["model"] == "glove"]
            if vest_dets and not glove_dets:
                summary["No Gloves"] = summary["vest_total"]

        return summary
