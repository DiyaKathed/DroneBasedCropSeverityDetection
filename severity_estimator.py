"""
severity_estimator.py
======================
Computer Vision-based Disease Severity & Visible Leaf Damage Assessment Module.

Uses color-space decomposition (HSV, LAB, Excess Green Index - ExG) and morphological
contour analysis to segment the visible leaf region and detect necrotic/chlorotic lesions.

Note:
This provides an experimental visible-damage estimate based on visible color anomalies
within segmented leaf boundaries. It is explicitly labeled:
"Experimental visible-damage estimate — not a validated agronomic severity diagnosis."
"""

import base64
import io
import cv2
import numpy as np
from PIL import Image


def segment_leaf(img_rgb: np.ndarray):
    """
    Segments the visible leaf region from the image using Excess Green Index (ExG),
    HSV color thresholds, and morphological closing to fill interior holes.
    
    Returns:
        leaf_mask: uint8 binary mask (255 for leaf pixels, 0 elsewhere)
        leaf_area_ratio: float (ratio of leaf pixels to total image pixels)
    """
    h, w = img_rgb.shape[:2]
    total_pixels = h * w
    
    # 1. Convert to float RGB for Excess Green Index (ExG = 2*G - R - B)
    img_float = img_rgb.astype(np.float32) / 255.0
    r = img_float[:, :, 0]
    g = img_float[:, :, 1]
    b = img_float[:, :, 2]
    
    denom = 2 * g + r + b + 1e-6
    norm_r = r / denom
    norm_g = g / denom
    norm_b = b / denom
    
    exg = 2 * norm_g - norm_r - norm_b
    exg_mask = (exg > 0.05).astype(np.uint8) * 255
    
    # 2. HSV color thresholds for plant foliage (greens, yellow-greens, olive-yellows)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h_channel = img_hsv[:, :, 0]
    s_channel = img_hsv[:, :, 1]
    v_channel = img_hsv[:, :, 2]
    
    # Green and yellowish plant tissue
    green_mask = cv2.inRange(img_hsv, (22, 30, 25), (95, 255, 255))
    
    # Brownish/yellowish foliage (e.g. wilted/diseased portions attached to leaf)
    foliage_mask = cv2.inRange(img_hsv, (10, 30, 20), (105, 255, 255))
    
    # Combine ExG and HSV foliage masks
    combined_raw = cv2.bitwise_or(exg_mask, green_mask)
    combined_raw = cv2.bitwise_or(combined_raw, foliage_mask)
    
    # Exclude extreme background values (very bright white/specular background or pure black borders)
    not_pure_white = ((r < 0.96) | (g < 0.96) | (b < 0.96)).astype(np.uint8) * 255
    not_pure_black = ((r > 0.04) | (g > 0.04) | (b > 0.04)).astype(np.uint8) * 255
    combined_raw = cv2.bitwise_and(combined_raw, not_pure_white)
    combined_raw = cv2.bitwise_and(combined_raw, not_pure_black)
    
    # 3. Morphological operations to remove noise and fill interior leaf holes
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    
    # Remove speckle noise
    opened = cv2.morphologyEx(combined_raw, cv2.MORPH_OPEN, kernel_small)
    # Bridge gaps across veins/lesions inside the leaf
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_large)
    
    # 4. Find dominant contours and fill their convex hulls or closed interiors
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    min_leaf_area = total_pixels * 0.03  # Leaf must be at least 3% of image
    
    significant_contours = [c for c in contours if cv2.contourArea(c) >= min_leaf_area]
    
    if not significant_contours:
        # Fallback to largest contour if > 1.5%
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= (total_pixels * 0.015):
                significant_contours = [largest]
    
    for cnt in significant_contours:
        cv2.drawContours(leaf_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    
    # Refine leaf mask by closing small interior holes (lesions inside leaf body)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel_large)
    
    leaf_pixels = int(np.sum(leaf_mask > 0))
    leaf_area_ratio = leaf_pixels / float(total_pixels)
    
    return leaf_mask, leaf_area_ratio, significant_contours


def detect_lesions(img_rgb: np.ndarray, leaf_mask: np.ndarray):
    """
    Detects visible lesion candidate regions (necrotic brown/black spots,
    chlorotic yellow patches, or powdery mildew) strictly within the segmented leaf mask.
    
    Returns:
        lesion_mask: uint8 binary mask (255 for lesion pixels, 0 elsewhere)
    """
    h, w = img_rgb.shape[:2]
    lesion_mask = np.zeros((h, w), dtype=np.uint8)
    
    if np.sum(leaf_mask > 0) == 0:
        return lesion_mask
    
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    
    h_chan = img_hsv[:, :, 0]
    s_chan = img_hsv[:, :, 1]
    v_chan = img_hsv[:, :, 2]
    
    l_chan = img_lab[:, :, 0]
    a_chan = img_lab[:, :, 1]
    b_chan = img_lab[:, :, 2]
    
    # 1. Necrotic spots: Dark brown/black/burnt patches
    # HSV: Hue in red/orange/brown [0, 22] or [165, 180], moderate saturation, lower brightness
    necrotic_hsv1 = cv2.inRange(img_hsv, (0, 35, 20), (22, 255, 140))
    necrotic_hsv2 = cv2.inRange(img_hsv, (160, 35, 20), (180, 255, 140))
    necrotic_hsv = cv2.bitwise_or(necrotic_hsv1, necrotic_hsv2)
    
    # LAB: positive a* (redder) with lower L (darker)
    necrotic_lab = ((a_chan > 132) & (l_chan < 130)).astype(np.uint8) * 255
    necrotic = cv2.bitwise_or(necrotic_hsv, necrotic_lab)
    
    # 2. Chlorotic lesions: Yellow/yellow-orange blight/rust patches
    # Yellowish hues with distinct saturation
    chlorotic_hsv = cv2.inRange(img_hsv, (15, 60, 80), (32, 255, 240))
    # LAB b* channel (yellows) distinctly higher than greens
    chlorotic_lab = ((b_chan > 155) & (a_chan > 120) & (l_chan > 80)).astype(np.uint8) * 255
    chlorotic = cv2.bitwise_or(chlorotic_hsv, chlorotic_lab)
    
    # 3. Very dark necrotic/fungal spot centers (black spots, septoria, anthracnose)
    dark_spots = ((v_chan < 60) & (s_chan > 20)).astype(np.uint8) * 255
    
    # 4. Powdery mildew / greyish fungal patches (high lightness, low saturation inside leaf)
    powdery = ((l_chan > 185) & (s_chan < 40) & (v_chan > 170)).astype(np.uint8) * 255
    
    # Combine candidate lesion masks
    candidate_lesions = cv2.bitwise_or(necrotic, chlorotic)
    candidate_lesions = cv2.bitwise_or(candidate_lesions, dark_spots)
    candidate_lesions = cv2.bitwise_or(candidate_lesions, powdery)
    
    # Strictly restrict to segmented leaf interior
    candidate_lesions = cv2.bitwise_and(candidate_lesions, leaf_mask)
    
    # Filter out single-pixel noise and smooth lesion clusters
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lesion_cleaned = cv2.morphologyEx(candidate_lesions, cv2.MORPH_OPEN, kernel_clean)
    
    # Small closing to connect adjacent lesion spots
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lesion_mask = cv2.morphologyEx(lesion_cleaned, cv2.MORPH_CLOSE, kernel_close)
    
    # Final boundary clamp to leaf mask
    lesion_mask = cv2.bitwise_and(lesion_mask, leaf_mask)
    
    return lesion_mask


def create_severity_overlay(
    img_rgb: np.ndarray,
    leaf_mask: np.ndarray,
    lesion_mask: np.ndarray,
    affected_pct: float,
    severity_level: str
) -> str:
    """
    Generates a visual diagnostic overlay:
    - Segmented leaf boundary outlined in neon emerald green.
    - Detected lesion regions highlighted with a vivid semi-transparent crimson overlay.
    - Information banner at the bottom showing affected area and experimental status.
    
    Returns base64 data URI string ("data:image/png;base64,...").
    """
    h, w = img_rgb.shape[:2]
    
    # Work on a copy in BGR for OpenCV drawing
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    overlay = img_bgr.copy()
    
    # 1. Tint lesion regions with vibrant crimson (BGR: [30, 30, 230])
    lesion_indices = lesion_mask > 0
    overlay[lesion_indices] = [30, 30, 230]
    
    # Blend overlay with original (alpha = 0.55 original, 0.45 tint)
    blended = cv2.addWeighted(overlay, 0.45, img_bgr, 0.55, 0)
    
    # 2. Outline leaf boundary with emerald green (BGR: [60, 220, 80])
    leaf_contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, leaf_contours, -1, (60, 220, 80), 2)
    
    # 3. Outline lesion contours with bright orange-red (BGR: [0, 100, 255])
    lesion_contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, lesion_contours, -1, (0, 100, 255), 1)
    
    # 4. Add HUD info banner at the bottom
    banner_h = max(36, int(h * 0.11))
    banner_y = h - banner_h
    
    # Semi-transparent dark banner background
    sub_img = blended[banner_y:h, 0:w]
    dark_rect = np.full_like(sub_img, (15, 23, 42))  # Slate dark #0f172a
    blended[banner_y:h, 0:w] = cv2.addWeighted(dark_rect, 0.85, sub_img, 0.15, 0)
    
    # Text in banner
    font_scale = max(0.40, min(0.65, w / 600.0))
    thickness = 1
    
    text_line1 = f"Visible Damage: {affected_pct:.1f}% leaf area ({severity_level})"
    text_line2 = "Leaf Boundary (Green) | Lesions (Red) | Experimental CV Estimate"
    
    y_pos1 = banner_y + int(banner_h * 0.45)
    y_pos2 = banner_y + int(banner_h * 0.85)
    
    # Level-based text color
    if severity_level == "Minimal / Healthy":
        badge_color = (120, 230, 120)  # Green
    elif severity_level == "Mild":
        badge_color = (80, 210, 240)   # Yellow
    elif severity_level == "Moderate":
        badge_color = (50, 140, 255)   # Orange
    else:
        badge_color = (80, 80, 255)    # Red
        
    cv2.putText(blended, text_line1, (12, y_pos1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, badge_color, thickness, cv2.LINE_AA)
    cv2.putText(blended, text_line2, (12, y_pos2), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.82, (180, 190, 205), thickness, cv2.LINE_AA)
    
    # Convert blended BGR to RGB and encode to PNG
    result_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(result_rgb)
    
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG", optimize=True)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return f"data:image/png;base64,{img_str}"


def classify_severity_thresholds(affected_pct: float) -> tuple[str, str]:
    """
    Standard phytopathological visual assessment scale
    (adapted from Horsfall-Barratt & EPPO standard area diagrams).
    
    Returns (level_name, description).
    """
    if affected_pct < 3.0:
        return (
            "Minimal / Healthy",
            "Less than 3% of the segmented leaf area exhibits visible lesions or discoloration."
        )
    elif affected_pct < 10.0:
        return (
            "Mild",
            "Between 3% and 10% of visible leaf area displays chlorotic or necrotic spots."
        )
    elif affected_pct < 25.0:
        return (
            "Moderate",
            "Between 10% and 25% of visible leaf area is covered by disease lesions."
        )
    else:
        return (
            "Severe",
            "Over 25% of visible leaf area is compromised by extensive lesions or tissue necrosis."
        )


def estimate_severity(img_input):
    """
    Full pipeline for estimating visible plant disease severity.
    
    Args:
        img_input: PIL.Image, file path (str), or numpy array (RGB)
        
    Returns:
        dict containing structured severity assessment data.
    """
    try:
        # Load image into numpy RGB
        if isinstance(img_input, str):
            with Image.open(img_input) as p_img:
                img_rgb = np.array(p_img.convert("RGB"))
        elif isinstance(img_input, Image.Image):
            img_rgb = np.array(img_input.convert("RGB"))
        elif isinstance(img_input, np.ndarray):
            if img_input.ndim == 2:
                img_rgb = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
            elif img_input.shape[2] == 4:
                img_rgb = cv2.cvtColor(img_input, cv2.COLOR_RGBA2RGB)
            else:
                img_rgb = img_input.copy()
        else:
            return {
                "status": "Failed",
                "level": None,
                "affected_area_percentage": None,
                "explanation": "Unsupported image format provided.",
                "method": "Computer Vision Segmentation",
                "overlay_image": None,
                "disclaimer": "Experimental visible-damage estimate — not a validated agronomic severity diagnosis."
            }

        # Check for minimum image dimensions
        h, w = img_rgb.shape[:2]
        if h < 32 or w < 32:
            return {
                "status": "Failed",
                "level": None,
                "affected_area_percentage": None,
                "explanation": "Image dimensions are too small for reliable leaf segmentation.",
                "method": "Computer Vision Segmentation",
                "overlay_image": None,
                "disclaimer": "Experimental visible-damage estimate — not a validated agronomic severity diagnosis."
            }

        # 1. Segment leaf region
        leaf_mask, leaf_area_ratio, contours = segment_leaf(img_rgb)
        leaf_pixel_count = int(np.sum(leaf_mask > 0))
        
        # If no significant leaf area was segmented (< 2.5% of image)
        if leaf_pixel_count < (h * w * 0.025):
            return {
                "status": "Unable to segment leaf",
                "level": None,
                "affected_area_percentage": None,
                "leaf_area_ratio": round(leaf_area_ratio * 100, 2),
                "explanation": (
                    "Could not detect a clearly defined leaf region in this image. "
                    "Ensure the leaf is clearly visible against the background."
                ),
                "method": "Color-Space Adaptive Segmentation (HSV/ExG)",
                "overlay_image": None,
                "disclaimer": "Experimental visible-damage estimate — not a validated agronomic severity diagnosis."
            }

        # 2. Detect candidate visible lesions within the leaf
        lesion_mask = detect_lesions(img_rgb, leaf_mask)
        lesion_pixel_count = int(np.sum(lesion_mask > 0))
        
        affected_pct = (lesion_pixel_count / float(leaf_pixel_count)) * 100.0
        affected_pct = round(min(100.0, max(0.0, affected_pct)), 2)
        
        # 3. Classify into documented visual damage levels
        level, level_description = classify_severity_thresholds(affected_pct)
        
        # 4. Generate visual overlay
        overlay_b64 = create_severity_overlay(
            img_rgb, leaf_mask, lesion_mask, affected_pct, level
        )
        
        explanation = (
            f"Estimated {affected_pct:.1f}% visible lesion coverage across the detected leaf area "
            f"({leaf_pixel_count:,} pixels, {leaf_area_ratio * 100:.1f}% of image). {level_description}"
        )

        return {
            "status": "Assessed",
            "level": level,
            "affected_area_percentage": affected_pct,
            "leaf_coverage_percentage": round(leaf_area_ratio * 100, 2),
            "explanation": explanation,
            "method": "Computer Vision Adaptive Color-Space Segmentation (HSV, LAB, ExG)",
            "overlay_image": overlay_b64,
            "thresholds_used": {
                "Minimal / Healthy": "< 3%",
                "Mild": "3% - 10%",
                "Moderate": "10% - 25%",
                "Severe": "> 25%"
            },
            "disclaimer": "Experimental visible-damage estimate — not a validated agronomic severity diagnosis."
        }

    except Exception as exc:
        return {
            "status": "Error",
            "level": None,
            "affected_area_percentage": None,
            "explanation": f"Severity analysis failed: {str(exc)}",
            "method": "Computer Vision Segmentation",
            "overlay_image": None,
            "disclaimer": "Experimental visible-damage estimate — not a validated agronomic severity diagnosis."
        }


if __name__ == "__main__":
    import sys
    test_path = sys.argv[1] if len(sys.argv) > 1 else "test_leaf3.jpeg"
    print(f"Testing severity estimation on: {test_path}")
    result = estimate_severity(test_path)
    print("Status:", result["status"])
    print("Level:", result["level"])
    print("Affected Area %:", result.get("affected_area_percentage"))
    print("Leaf Area %:", result.get("leaf_coverage_percentage"))
    print("Explanation:", result["explanation"])
    print("Has Overlay:", result["overlay_image"] is not None)
    if result["overlay_image"]:
        print("Overlay string length:", len(result["overlay_image"]))
