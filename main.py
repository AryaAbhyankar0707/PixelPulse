#!/usr/bin/env python3
"""
PixelPulse -- Intelligent Image Health & Quality Analyzer
============================================================
Command-line entry point. See README.md for full usage documentation.

Usage
-----
    python main.py --input path/to/image.jpg
    python main.py --input path/to/image.jpg --analyze
    python main.py --input path/to/image.jpg --enhance
    python main.py --input path/to/image.jpg --report
    python main.py --compare original.jpg enhanced.jpg
"""

import argparse
import os
import sys

import cv2

from core.image_loader import load_image, ImageLoadError
from core.preprocessing import resize_for_analysis
from core import quality_engine, recommendation_engine
from analysis import (
    histogram_analysis,
    noise_analysis,
    blur_analysis,
    edge_analysis,
    texture_analysis,
    feature_analysis,
)
from enhancement import denoise, sharpen, contrast as contrast_enhance
from reports import report_generator


def _analyze_arrays(gray, bgr, include_features=True):
    """Run the PixelPulse analysis pipeline directly on image arrays."""
    gray_resized, _bgr_resized, _scale = resize_for_analysis(gray, bgr)

    analysis_results = {
        "histogram": histogram_analysis.analyze(gray_resized),
        "noise": noise_analysis.analyze(gray_resized),
        "blur": blur_analysis.analyze(gray_resized),
        "edge": edge_analysis.analyze(gray_resized),
        "texture": texture_analysis.analyze(gray_resized),
    }
    if include_features:
        analysis_results["features"] = feature_analysis.analyze(gray_resized)

    quality_report = quality_engine.build_quality_report(analysis_results)
    problems = recommendation_engine.diagnose(analysis_results)
    return analysis_results, quality_report, problems


def run_analysis(path, include_features=True):
    """Load an image and run the full PixelPulse analysis pipeline."""
    loaded = load_image(path)
    analysis_results, quality_report, problems = _analyze_arrays(
        loaded.gray, loaded.bgr, include_features=include_features
    )
    return loaded, analysis_results, quality_report, problems


def cmd_default_or_analyze(args):
    try:
        loaded, analysis_results, quality_report, problems = run_analysis(args.input)
    except ImageLoadError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    text_report = report_generator.render_text_report(
        image_name=os.path.basename(loaded.path),
        resolution=loaded.resolution_str,
        quality_report=quality_report,
        analysis_results=analysis_results,
        problems=problems,
    )
    print(text_report)

    if args.analyze:
        print("\nSub-scores (0-100):")
        for name, value in quality_report["sub_scores"].items():
            print(f"  {name:<18}: {value:6.2f}")
        if "features" in analysis_results:
            feats = analysis_results["features"]
            print("\nFeature/structure analysis:")
            print(f"  HOG vector length     : {feats['hog_vector_length']}")
            print(f"  HOG energy            : {feats['hog_energy']}")
            print(f"  SIFT keypoints        : {feats['sift_keypoint_count']}")
            print(f"  Feature stability     : {feats['feature_stability_label']}")


def cmd_enhance(args):
    try:
        loaded, analysis_results, quality_report, problems = run_analysis(args.input)
    except ImageLoadError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    actions = recommendation_engine.recommended_actions(problems)
    if not actions:
        print("No enhancement actions recommended -- image quality is already acceptable.")
        actions = []

    working = loaded.bgr.copy()
    applied = []
    rejected = []
    current_score = quality_report["score"]

    enhancement_functions = {
        "denoise": (denoise.apply, "denoise (Non-Local Means)"),
        "sharpen": (sharpen.apply, "sharpen (Unsharp Mask)"),
        "contrast": (contrast_enhance.apply, "contrast (CLAHE)"),
    }

    # Evaluate every proposed operation before accepting it. This makes the
    # enhancement stage self-checking instead of blindly chaining filters.
    for action in actions:
        function, label = enhancement_functions[action]
        candidate = function(working)
        candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
        _, candidate_quality, _ = _analyze_arrays(
            candidate_gray, candidate, include_features=False
        )
        candidate_score = candidate_quality["score"]

        if candidate_score > current_score + 0.25:
            working = candidate
            current_score = candidate_score
            applied.append(label)
        else:
            rejected.append(label)

    out_dir = report_generator.ensure_output_dir()
    out_path = os.path.join(out_dir, "enhanced_image.jpg")
    report_generator.save_image(working, out_path)

    print(f"Original quality score : {quality_report['score']:.0f} / 100 ({quality_report['category']})")
    print(f"Applied enhancements   : {', '.join(applied) if applied else 'none'}")
    if rejected:
        print(f"Rejected enhancements  : {', '.join(rejected)}")
    print(f"Enhanced quality score : {current_score:.0f} / 100 ({quality_engine.score_to_category(current_score)})")
    print(f"Enhanced image saved to: {out_path}")


def cmd_report(args):
    try:
        loaded, analysis_results, quality_report, problems = run_analysis(args.input)
    except ImageLoadError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    out_dir = report_generator.ensure_output_dir()
    image_name = os.path.basename(loaded.path)

    text_report = report_generator.render_text_report(
        image_name, loaded.resolution_str, quality_report, analysis_results, problems
    )
    json_report = report_generator.render_json_report(
        image_name, loaded.resolution_str, quality_report, analysis_results, problems
    )

    with open(os.path.join(out_dir, "analysis_report.txt"), "w", encoding="utf-8") as f:
        f.write(text_report)
    with open(os.path.join(out_dir, "quality_report.json"), "w", encoding="utf-8") as f:
        f.write(json_report)

    report_generator.save_histogram_plot(
        analysis_results["histogram"]["histogram"],
        os.path.join(out_dir, "histogram.png"),
    )
    report_generator.save_edge_map(
        analysis_results["edge"]["edge_map"],
        os.path.join(out_dir, "edge_map.png"),
    )

    print(text_report)
    print(f"\nSaved: {out_dir}/analysis_report.txt")
    print(f"Saved: {out_dir}/quality_report.json")
    print(f"Saved: {out_dir}/histogram.png")
    print(f"Saved: {out_dir}/edge_map.png")


def cmd_compare(args):
    try:
        _, _, quality_before, _ = run_analysis(args.compare[0], include_features=False)
        _, _, quality_after, _ = run_analysis(args.compare[1], include_features=False)
    except ImageLoadError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    before_score = quality_before["score"]
    after_score = quality_after["score"]
    improvement = round(after_score - before_score, 2)

    before_sub = quality_before["sub_scores"]
    after_sub = quality_after["sub_scores"]

    improved = [k for k in before_sub if after_sub[k] > before_sub[k] + 1e-6]
    degraded = [k for k in before_sub if after_sub[k] < before_sub[k] - 1e-6]

    lines = []
    lines.append("=" * 40)
    lines.append("PIXELPULSE COMPARISON REPORT".center(40))
    lines.append("=" * 40)
    lines.append(f"ORIGINAL")
    lines.append(f"Quality Score : {before_score:.0f}")
    lines.append("")
    lines.append(f"ENHANCED")
    lines.append(f"Quality Score : {after_score:.0f}")
    lines.append("")
    lines.append(f"Improvement   : {'+' if improvement >= 0 else ''}{improvement:.0f}")
    lines.append("")
    lines.append("Improved:")
    for k in improved:
        lines.append(f"  + {k}")
    lines.append("")
    lines.append("Degraded:")
    for k in degraded:
        lines.append(f"  - {k}")
    lines.append("=" * 40)

    report_text = "\n".join(lines)
    print(report_text)

    out_dir = report_generator.ensure_output_dir()
    with open(os.path.join(out_dir, "comparison_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nSaved: {out_dir}/comparison_report.txt")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="PixelPulse",
        description="Intelligent Image Health & Quality Analyzer",
    )
    parser.add_argument("--input", help="Path to the input image to analyze")
    parser.add_argument("--analyze", action="store_true",
                         help="Show detailed sub-scores and feature/structure analysis")
    parser.add_argument("--enhance", action="store_true",
                         help="Diagnose issues and apply only the warranted enhancements")
    parser.add_argument("--report", action="store_true",
                         help="Generate full text/JSON reports plus histogram and edge map images")
    parser.add_argument("--compare", nargs=2, metavar=("ORIGINAL", "ENHANCED"),
                         help="Compare quality scores of two images (before vs after)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.compare:
        cmd_compare(args)
        return

    if not args.input:
        parser.error("--input is required unless using --compare")

    if args.enhance:
        cmd_enhance(args)
    elif args.report:
        cmd_report(args)
    else:
        cmd_default_or_analyze(args)


if __name__ == "__main__":
    main()
