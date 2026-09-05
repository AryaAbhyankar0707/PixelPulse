"""
Tests for core.quality_engine and core.recommendation_engine.
"""

from config import settings
from core import quality_engine, recommendation_engine


def _fake_sub_scores(**overrides):
    base = {name: 80.0 for name in settings.SCORE_WEIGHTS}
    base.update(overrides)
    return base


def test_compute_score_all_perfect_is_100():
    perfect = {name: 100.0 for name in settings.SCORE_WEIGHTS}
    assert quality_engine.compute_score(perfect) == 100.0


def test_compute_score_all_zero_is_0():
    zero = {name: 0.0 for name in settings.SCORE_WEIGHTS}
    assert quality_engine.compute_score(zero) == 0.0


def test_compute_score_is_weighted_average():
    scores = _fake_sub_scores(sharpness=100.0, texture=0.0)
    result = quality_engine.compute_score(scores)
    # Should sit strictly between 0 and 100, and below the "all 80" baseline
    # only slightly since texture has the smallest weight.
    assert 0.0 < result < 100.0


def test_score_to_category_bands():
    assert quality_engine.score_to_category(10) == "POOR"
    assert quality_engine.score_to_category(45) == "WEAK"
    assert quality_engine.score_to_category(65) == "ACCEPTABLE"
    assert quality_engine.score_to_category(80) == "GOOD"
    assert quality_engine.score_to_category(95) == "EXCELLENT"


def test_rank_weaknesses_surfaces_lowest_score():
    scores = _fake_sub_scores(sharpness=5.0)
    weakest = quality_engine.rank_weaknesses(scores, top_n=1)
    assert weakest[0][0] == "sharpness"
    assert weakest[0][1] == 5.0


def test_build_quality_report_shape():
    fake_analysis = {
        "blur": {"sharpness_score": 90.0, "laplacian_variance": 400.0, "blur_label": "SHARP"},
        "histogram": {
            "contrast_score": 85.0, "contrast_std": 55.0, "contrast_label": "GOOD",
            "brightness_score": 95.0, "brightness_mean": 130.0, "brightness_label": "NORMAL",
        },
        "noise": {"noise_score": 98.0, "noise_sigma": 0.5, "noise_label": "LOW"},
        "edge": {"edge_score": 80.0, "edge_density": 0.09, "edge_label": "GOOD"},
        "texture": {"texture_score": 70.0, "texture_energy": 18.0, "texture_label": "MODERATE"},
    }
    report = quality_engine.build_quality_report(fake_analysis)

    assert set(report.keys()) == {"score", "category", "sub_scores", "weaknesses"}
    assert 0 <= report["score"] <= 100
    assert report["category"] in {"POOR", "WEAK", "ACCEPTABLE", "GOOD", "EXCELLENT"}
    assert len(report["sub_scores"]) == len(settings.SCORE_WEIGHTS)


def test_recommendation_engine_flags_blur_and_noise():
    fake_analysis = {
        "blur": {"sharpness_score": 10.0, "laplacian_variance": 20.0, "blur_label": "SEVERE"},
        "histogram": {
            "contrast_score": 80.0, "contrast_std": 60.0, "contrast_label": "GOOD",
            "brightness_score": 90.0, "brightness_mean": 130.0, "brightness_label": "NORMAL",
        },
        "noise": {"noise_score": 20.0, "noise_sigma": 15.0, "noise_label": "HIGH"},
        "edge": {"edge_score": 40.0, "edge_density": 0.03, "edge_label": "MODERATE"},
        "texture": {"texture_score": 60.0, "texture_energy": 16.0, "texture_label": "MODERATE"},
    }
    problems = recommendation_engine.diagnose(fake_analysis)
    issues = [p["issue"] for p in problems]

    assert any("blur" in i.lower() for i in issues)
    assert any("noise" in i.lower() for i in issues)

    actions = recommendation_engine.recommended_actions(problems)
    assert "sharpen" in actions
    assert "denoise" in actions


def test_recommendation_engine_no_problems_for_clean_image():
    fake_analysis = {
        "blur": {"sharpness_score": 95.0, "laplacian_variance": 500.0, "blur_label": "SHARP"},
        "histogram": {
            "contrast_score": 90.0, "contrast_std": 60.0, "contrast_label": "GOOD",
            "brightness_score": 98.0, "brightness_mean": 128.0, "brightness_label": "NORMAL",
        },
        "noise": {"noise_score": 99.0, "noise_sigma": 0.2, "noise_label": "LOW"},
        "edge": {"edge_score": 90.0, "edge_density": 0.1, "edge_label": "GOOD"},
        "texture": {"texture_score": 90.0, "texture_energy": 25.0, "texture_label": "RICH"},
    }
    problems = recommendation_engine.diagnose(fake_analysis)
    assert problems == []
    assert recommendation_engine.recommended_actions(problems) == []
