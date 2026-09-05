"""
reports.report_generator
--------------------------
Renders the results of a PixelPulse run into human-readable (text) and
machine-readable (JSON) reports, plus optional saved visual artifacts
(histogram plot, edge map, enhanced image).
"""

import json
import os
from datetime import datetime

import cv2
import numpy as np


def _bar(width=40):
    return "=" * width


def render_text_report(image_name, resolution, quality_report, analysis_results, problems):
    score = quality_report["score"]
    category = quality_report["category"]
    hist = analysis_results["histogram"]
    noise = analysis_results["noise"]
    blur = analysis_results["blur"]
    edge = analysis_results["edge"]
    texture = analysis_results["texture"]

    lines = []
    lines.append(_bar())
    lines.append("PIXELPULSE IMAGE REPORT".center(40))
    lines.append(_bar())
    lines.append(f"Image           : {image_name}")
    lines.append(f"Resolution      : {resolution}")
    lines.append("")
    lines.append(f"Overall Quality : {score:.0f} / 100")
    lines.append(f"Quality Level   : {category}")
    lines.append("")
    lines.append("-" * 40)
    lines.append("ANALYSIS")
    lines.append("-" * 40)
    lines.append(f"Noise             : {noise['noise_label']}")
    lines.append(f"Blur              : {blur['blur_label']}")
    lines.append(f"Contrast          : {hist['contrast_label']}")
    lines.append(f"Brightness        : {hist['brightness_label']}")
    lines.append(f"Edge Preservation : {edge['edge_label']}")
    lines.append(f"Texture Quality   : {texture['texture_label']}")
    lines.append("")

    if problems:
        lines.append("-" * 40)
        lines.append("DETECTED ISSUES")
        lines.append("-" * 40)
        for i, p in enumerate(problems, start=1):
            lines.append(f"{i}. {p['issue']} (evidence: {p['evidence']})")
        lines.append("")

        actions = {p["action"] for p in problems if p["action"]}
        if actions:
            lines.append("-" * 40)
            lines.append("RECOMMENDATION")
            lines.append("-" * 40)
            lines.append("Apply:")
            action_text = {
                "sharpen": "Unsharp-mask sharpening",
                "denoise": "Non-local means denoising",
                "contrast": "CLAHE contrast/exposure correction",
            }
            for a in actions:
                lines.append(f"  -> {action_text.get(a, a)}")
            lines.append("")
    else:
        lines.append("No significant issues detected.")
        lines.append("")

    lines.append(_bar())
    return "\n".join(lines)


def render_json_report(image_name, resolution, quality_report, analysis_results, problems):
    def strip_arrays(d):
        """Drop non-serializable numpy arrays (histograms, edge maps) from
        the analysis dicts before JSON-dumping; those are saved separately
        as image files."""
        clean = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                continue
            clean[k] = v
        return clean

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "image": image_name,
        "resolution": resolution,
        "overall_score": quality_report["score"],
        "quality_category": quality_report["category"],
        "sub_scores": quality_report["sub_scores"],
        "weaknesses": [
            {"metric": name, "score": s, "weight": w}
            for name, s, w in quality_report["weaknesses"]
        ],
        "analysis": {
            "histogram": strip_arrays(analysis_results["histogram"]),
            "noise": strip_arrays(analysis_results["noise"]),
            "blur": strip_arrays(analysis_results["blur"]),
            "edge": strip_arrays(analysis_results["edge"]),
            "texture": strip_arrays(analysis_results["texture"]),
            "features": strip_arrays(analysis_results.get("features", {})),
        },
        "detected_issues": problems,
    }
    return json.dumps(payload, indent=2)


def save_histogram_plot(histogram, out_path):
    """Save a simple bar-style histogram plot without requiring matplotlib,
    by rendering directly to an image array with OpenCV so the project has
    no heavyweight plotting dependency."""
    width, height = 512, 256
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    hist = histogram / (histogram.max() + 1e-8)
    bin_width = max(1, width // len(hist))

    for i, value in enumerate(hist):
        x = i * bin_width
        bar_height = int(value * (height - 10))
        cv2.rectangle(
            canvas,
            (x, height - bar_height),
            (x + bin_width - 1, height),
            (60, 60, 60),
            thickness=-1,
        )

    cv2.imwrite(out_path, canvas)


def save_edge_map(edge_map, out_path):
    cv2.imwrite(out_path, edge_map)


def save_image(bgr, out_path):
    cv2.imwrite(out_path, bgr)


def ensure_output_dir(path="outputs"):
    os.makedirs(path, exist_ok=True)
    return path
