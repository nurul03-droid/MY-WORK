"""
Blueprint land-map GeoAI detection pipeline
=============================================
Detects plot/parcel boundaries from scanned line-drawing style land blueprints
(as opposed to satellite/drone imagery, which needs a different segmentation approach).

Pipeline: preprocess -> Hough line detection -> close gaps -> extract contours
          -> simplify to polygons -> OCR-label matching -> georeference -> validate

Install:
    pip install opencv-python shapely pytesseract
    system binary: sudo apt install tesseract-ocr   (or brew install tesseract on Mac)
"""

import argparse
import json

import cv2
import numpy as np
import pytesseract
from shapely.geometry import Polygon
from shapely.affinity import scale


# ---------------------------------------------------------------------------
# Step 1 - Preprocessing
# ---------------------------------------------------------------------------

def preprocess_blueprint(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    denoised = cv2.fastNlMeansDenoising(img, h=10)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


# ---------------------------------------------------------------------------
# Step 2 - Line detection (Hough Transform)
# ---------------------------------------------------------------------------

def detect_lines(binary_img: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(binary_img, 50, 150)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=50,
        minLineLength=30, maxLineGap=10
    )
    line_mask = np.zeros_like(binary_img)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=2)
    return line_mask


# ---------------------------------------------------------------------------
# Step 3 - Close gaps and extract closed plot contours
# ---------------------------------------------------------------------------

def close_and_extract_contours(line_mask: np.ndarray, min_area: int = 300):
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) > min_area]


# ---------------------------------------------------------------------------
# Step 4 - Simplify into clean straight-edge polygons
# ---------------------------------------------------------------------------

def simplify_to_polygons(contours) -> list:
    polygons = []
    for c in contours:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) >= 3:
            pts = [tuple(p[0]) for p in approx]
            poly = Polygon(pts)
            if poly.is_valid and poly.area > 0:
                polygons.append(poly)
    return polygons


# ---------------------------------------------------------------------------
# Step 5 - OCR plot numbers/labels and match to nearest polygon
# ---------------------------------------------------------------------------

def extract_labels(binary_img: np.ndarray, polygons: list) -> list:
    data = pytesseract.image_to_data(binary_img, output_type=pytesseract.Output.DICT)
    labeled = []
    for poly in polygons:
        cx, cy = poly.centroid.x, poly.centroid.y
        best_text, best_dist = None, float("inf")
        for i, text in enumerate(data["text"]):
            if not text.strip():
                continue
            tx = data["left"][i] + data["width"][i] / 2
            ty = data["top"][i] + data["height"][i] / 2
            dist = (tx - cx) ** 2 + (ty - cy) ** 2
            if dist < best_dist:
                best_dist, best_text = dist, text.strip()
        labeled.append({"polygon": poly, "label": best_text})
    return labeled


# ---------------------------------------------------------------------------
# Step 6 - Georeference: convert pixel coords to metric coords via a scale bar
# ---------------------------------------------------------------------------

def pixels_to_metric(polygon: Polygon, pixel_to_meter_ratio: float) -> Polygon:
    """Use this when the blueprint has only a scale bar (e.g. '1 cm = 5 m') and no
    real-world GPS tie-in. For true lat/long georeferencing with GCPs instead,
    use GDAL as shown in the main pipeline guide."""
    return scale(polygon, xfact=pixel_to_meter_ratio, yfact=pixel_to_meter_ratio, origin=(0, 0))


# ---------------------------------------------------------------------------
# Step 7 - Validation
# ---------------------------------------------------------------------------

def validate(polygon: Polygon, declared_area_sqm: float | None, tolerance: float = 0.10) -> dict:
    measured = polygon.area
    flagged = False
    reason = None
    if not polygon.is_valid:
        flagged, reason = True, "Invalid/self-intersecting geometry"
    elif declared_area_sqm:
        diff_ratio = abs(measured - declared_area_sqm) / declared_area_sqm
        if diff_ratio > tolerance:
            flagged, reason = True, f"Area mismatch: {diff_ratio:.1%} difference from declared record"
    return {"measured_area_sqm": measured, "flagged": flagged, "flag_reason": reason}


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def run_blueprint_pipeline(image_path: str, pixel_to_meter_ratio: float = 1.0):
    binary = preprocess_blueprint(image_path)
    line_mask = detect_lines(binary)
    contours = close_and_extract_contours(line_mask)
    polygons = simplify_to_polygons(contours)
    labeled = extract_labels(binary, polygons)

    results = []
    for item in labeled:
        metric_poly = pixels_to_metric(item["polygon"], pixel_to_meter_ratio)
        validation = validate(metric_poly, declared_area_sqm=None)  # fill in after matching to OCR'd RoR record
        results.append({
            "label": item["label"],
            "polygon": metric_poly,
            "measured_area_sqm": validation["measured_area_sqm"],
            "flagged": validation["flagged"],
            "flag_reason": validation["flag_reason"],
        })

    print(f"Detected {len(results)} plot(s) on the blueprint.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blueprint land-map GeoAI detection pipeline")
    parser.add_argument("image", help="Path to a scanned blueprint/land-map image")
    parser.add_argument("--scale", type=float, default=1.0,
                         help="pixel-to-meter ratio, e.g. 0.05 if 1 pixel = 5 cm")
    args = parser.parse_args()

    results = run_blueprint_pipeline(args.image, pixel_to_meter_ratio=args.scale)

    for i, r in enumerate(results):
        print(json.dumps({
            "id": i,
            "label": r["label"],
            "area_sqm": round(r["measured_area_sqm"], 2),
            "flagged": r["flagged"],
        }, indent=2))
