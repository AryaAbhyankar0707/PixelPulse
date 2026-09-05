"""
core.quality_engine
---------------------
The PixelPulse Quality Score Engine (Module 3): combines the individual
analysis sub-scores into one weighted, explainable 0-100 quality score
and category label, and surfaces the dominant reason(s) behind the score.

This is the module that turns a bag of independent measurements into a
single coherent verdict -- the "genuinely interesting part" of the system.
"""

from config import settings


def compute_score(sub_scores):
    """
    sub_scores: dict with keys matching config.settings.SCORE_WEIGHTS
                ('sharpness', 'contrast', 'noise', 'illumination',
                 'edge_preservation', 'texture'), each a 0-100 float.

    Returns the weighted overall score (float, 0-100).
    """
    total = 0.0
    for key, weight in settings.SCORE_WEIGHTS.items():
        total += sub_scores.get(key, 0.0) * weight
    return round(total, 2)


def score_to_category(score):
    for low, high, label in settings.QUALITY_BANDS:
        if low <= score < high:
            return label
    return "EXCELLENT" if score >= 100 else "POOR"


def rank_weaknesses(sub_scores, top_n=2):
    """
    Return the `top_n` lowest-scoring sub-metrics, weighted by their
    contribution to the overall score, so that a low-weight-but-terrible
    metric and a high-weight-but-mediocre metric can both surface when
    relevant. Returns a list of (metric_name, sub_score, weight) tuples,
    worst first.
    """
    weighted = [
        (name, sub_scores.get(name, 0.0), settings.SCORE_WEIGHTS[name])
        for name in settings.SCORE_WEIGHTS
    ]
    # Sort by (sub_score) ascending primarily -- a very low raw score is a
    # real problem regardless of weight -- with weight as a tiebreaker so
    # that among similarly bad scores, the higher-weighted one leads.
    weighted.sort(key=lambda item: (item[1], -item[2]))
    return weighted[:top_n]


def build_quality_report(analysis_results):
    """
    analysis_results: dict produced by main.run_analysis(), containing the
    raw output dicts of each analysis module.

    Returns a dict: {
        'score': float,
        'category': str,
        'sub_scores': dict,
        'weaknesses': [(name, score, weight), ...]
    }
    """
    sub_scores = {
        "sharpness": analysis_results["blur"]["sharpness_score"],
        "contrast": analysis_results["histogram"]["contrast_score"],
        "noise": analysis_results["noise"]["noise_score"],
        "illumination": analysis_results["histogram"]["brightness_score"],
        "edge_preservation": analysis_results["edge"]["edge_score"],
        "texture": analysis_results["texture"]["texture_score"],
    }

    score = compute_score(sub_scores)
    category = score_to_category(score)
    weaknesses = rank_weaknesses(sub_scores)

    return {
        "score": score,
        "category": category,
        "sub_scores": sub_scores,
        "weaknesses": weaknesses,
    }
