"""
core.recommendation_engine
-----------------------------
Module 4 -- Enhancement Recommendation. Rather than blindly applying every
enhancement filter available, this module inspects the analysis results and
proposes only the corrections that are actually warranted, with the
evidence behind each recommendation. --enhance uses this module's output to
decide which of enhancement/{denoise,sharpen,contrast}.py to actually run.
"""

from config import settings


def diagnose(analysis_results):
    """
    Inspect raw analysis results and return a list of problem dicts:
        {'issue': str, 'severity': str, 'evidence': str, 'action': str}
    ordered roughly by severity (worst first).
    """
    problems = []

    blur = analysis_results["blur"]
    if blur["blur_label"] in ("SEVERE", "MODERATE", "MILD"):
        problems.append({
            "issue": f"{blur['blur_label'].title()} blur / low sharpness",
            "severity": blur["blur_label"],
            "evidence": f"Laplacian variance = {blur['laplacian_variance']}",
            "action": "sharpen",
        })

    noise = analysis_results["noise"]
    if noise["noise_label"] in ("HIGH", "SEVERE", "MODERATE"):
        problems.append({
            "issue": f"{noise['noise_label'].title()} sensor/compression noise",
            "severity": noise["noise_label"],
            "evidence": f"Estimated noise sigma = {noise['noise_sigma']}",
            "action": "denoise",
        })

    hist = analysis_results["histogram"]
    if hist["contrast_label"] == "LOW":
        problems.append({
            "issue": "Low contrast",
            "severity": "MODERATE",
            "evidence": f"Histogram std = {hist['contrast_std']}",
            "action": "contrast",
        })

    if hist["brightness_label"] == "OVEREXPOSED":
        problems.append({
            "issue": "Overexposure",
            "severity": "MODERATE",
            "evidence": f"Mean brightness = {hist['brightness_mean']}",
            "action": "contrast",
        })
    elif hist["brightness_label"] == "DARK":
        problems.append({
            "issue": "Underexposure",
            "severity": "MODERATE",
            "evidence": f"Mean brightness = {hist['brightness_mean']}",
            "action": "contrast",
        })

    texture = analysis_results["texture"]
    if texture["texture_label"] == "FLAT":
        problems.append({
            "issue": "Flat / low-texture regions (possible over-smoothing or compression loss)",
            "severity": "MILD",
            "evidence": f"Texture energy = {texture['texture_energy']}",
            "action": None,  # informational only, no direct filter fixes this
        })

    severity_rank = {"SEVERE": 0, "HIGH": 1, "MODERATE": 2, "MILD": 3, "LOW": 4}
    problems.sort(key=lambda p: severity_rank.get(p["severity"], 5))
    return problems


def recommended_actions(problems):
    """Deduplicated, ordered list of enhancement action names to actually run."""
    seen = []
    for p in problems:
        if p["action"] and p["action"] not in seen:
            seen.append(p["action"])
    return seen
