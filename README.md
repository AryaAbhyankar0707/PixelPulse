# PixelPulse — Intelligent Image Health & Quality Analyzer

PixelPulse is a command-line tool that examines a photograph, measures
several independent aspects of its visual quality using classical image
processing techniques (no deep learning, no pretrained models), combines
those measurements into a single explainable 0–100 quality score, diagnoses
the most likely problems, and — if asked — applies only the enhancements
that are actually warranted.

This README assumes no prior context about the project: follow it top to
bottom to get a working setup.

---

## 1. What it does

```
Input Image
     │
     ▼
Image Loading & Validation        (core/image_loader.py)
     │
     ▼
Preprocessing (resize for analysis)  (core/preprocessing.py)
     │
     ▼
┌─────────────────────────────────────────────┐
│              Analysis (Module 1 & 2)          │
│  histogram · noise · blur · edge · texture ·  │
│  HOG / SIFT features                          │
└─────────────────────┬─────────────────────────┘
                       ▼
        Quality Score Engine (Module 3)
                       ▼
        Recommendation Engine (Module 4)
                       ▼
              Text / JSON Report
```

Given `street.jpg`, PixelPulse produces a report such as:

```
========================================
        PIXELPULSE IMAGE REPORT
========================================
Image           : street.jpg
Resolution      : 640 x 480

Overall Quality : 41 / 100
Quality Level   : WEAK

----------------------------------------
ANALYSIS
----------------------------------------
Noise             : LOW
Blur              : MODERATE
Contrast          : LOW
Brightness        : NORMAL
Edge Preservation : WEAK
Texture Quality   : FLAT

----------------------------------------
DETECTED ISSUES
----------------------------------------
1. Moderate blur / low sharpness (evidence: Laplacian variance = 70.05)
2. Low contrast (evidence: Histogram std = 25.85)

----------------------------------------
RECOMMENDATION
----------------------------------------
Apply:
  -> CLAHE contrast/exposure correction
  -> Unsharp-mask sharpening
========================================
```

---

## 2. Environment setup

**Requirements:** Python 3.9 or newer, and `pip`.

```bash
# 1. Clone the repository
git clone https://github.com/<github-username>/PixelPulse.git
cd PixelPulse

# 2. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

`requirements.txt` pulls in:
- `opencv-python-headless` — image I/O and the core CV algorithms (Canny,
  Laplacian, CLAHE, Non-Local Means denoising, SIFT, box filters)
- `numpy` — array math
- `matplotlib` — saving the histogram plot for `--report`
- `scikit-image` — HOG descriptor computation
- `pytest` — running the test suite

No GPU, internet access, or pretrained model download is required at any
point.

---

## 3. Running PixelPulse

All commands are run from the repository root, with the virtual
environment activated.

### Analyze an image (default report + sub-scores)
```bash
python main.py --input data/sample_images/degraded_sample.jpg --analyze
```

### Basic report only (no sub-scores / feature breakdown)
```bash
python main.py --input data/sample_images/degraded_sample.jpg
```

### Diagnose and auto-enhance (only the warranted filters are applied)
```bash
python main.py --input data/sample_images/degraded_sample.jpg --enhance
```
Saves `outputs/enhanced_image.jpg` and prints the before/after quality
score.

### Generate full report artifacts (text + JSON + histogram + edge map)
```bash
python main.py --input data/sample_images/sharp_sample.jpg --report
```
Writes to the `outputs/` directory:
- `analysis_report.txt`
- `quality_report.json`
- `histogram.png`
- `edge_map.png`

### Compare two images (e.g. before vs. after enhancement)
```bash
python main.py --compare data/sample_images/degraded_sample.jpg outputs/enhanced_image.jpg
```
Writes `outputs/comparison_report.txt` showing the score delta and which
sub-metrics improved or degraded.

Two ready-to-use sample images are provided under `data/sample_images/` —
`sharp_sample.jpg` (clean, high-quality) and `degraded_sample.jpg`
(synthetically blurred and washed out) — so the tool can be exercised
immediately without supplying your own image.

Supported input formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`.

---

## 4. Running the tests

```bash
python -m pytest tests/ -v
```

The suite (20 tests) covers:
- `test_loader.py` — file validation and error handling (missing file,
  wrong extension, empty file, corrupt data, too-small image)
- `test_analysis.py` — directional correctness of each analysis module
  (e.g. a blurred image must score lower on sharpness than its sharp
  original; a noisy image must score lower on the noise metric than its
  clean original)
- `test_quality.py` — the weighted scoring formula, quality-band
  classification, weakness ranking, and the recommendation engine's
  diagnosis logic

---

## 5. Project structure

```
PixelPulse/
├── README.md
├── statement.md
├── requirements.txt
├── main.py                      # CLI entry point
│
├── config/
│   └── settings.py              # all weights & thresholds, in one place
│
├── core/
│   ├── image_loader.py          # Module: load & validate an image
│   ├── preprocessing.py         # resize-for-analysis, light denoise
│   ├── quality_engine.py        # Module 3: weighted score + explanation
│   └── recommendation_engine.py # Module 4: diagnosis -> actions
│
├── analysis/                    # Module 1 & 2: measurement functions
│   ├── histogram_analysis.py    # brightness / contrast
│   ├── noise_analysis.py        # noise sigma estimate
│   ├── blur_analysis.py         # variance of Laplacian
│   ├── edge_analysis.py         # Canny edge density
│   ├── texture_analysis.py      # local std-dev texture energy
│   └── feature_analysis.py      # HOG / SIFT (diagnostic, not scored)
│
├── enhancement/                 # only invoked when actually warranted
│   ├── denoise.py                # Non-Local Means
│   ├── sharpen.py                # Unsharp masking
│   └── contrast.py               # CLAHE
│
├── reports/
│   └── report_generator.py      # text/JSON report + histogram/edge-map plots
│
├── tests/
│   ├── test_loader.py
│   ├── test_analysis.py
│   └── test_quality.py
│
├── data/sample_images/          # sharp_sample.jpg, degraded_sample.jpg
└── outputs/                     # generated reports & images land here
```

---

## 6. Design notes (for the report / viva)

- **Why classical CV, not deep learning:** every score is traceable to a
  concrete, explainable measurement (a variance, a standard deviation, a
  pixel-density ratio) rather than an opaque model output — which is what
  makes the "Recommended: apply sharpening because Laplacian variance =
  84.6" style of output possible.
- **Weighted, explainable scoring:** `core/quality_engine.py` combines six
  independent sub-scores using fixed weights defined in
  `config/settings.py` (sharpness 25%, contrast 20%, noise 15%,
  illumination 15%, edge preservation 15%, texture 10%), and separately
  reports which sub-metrics dragged the score down the most.
- **Diagnose before you fix:** `core/recommendation_engine.py` inspects
  the analysis output and only recommends (and `--enhance` only applies)
  the specific corrective filter each detected problem calls for, instead
  of running every enhancement on every image.
- **HOG/SIFT are diagnostic, not scored:** they describe *what kind* of
  content is present (how many stable, matchable features it has) rather
  than *how good* the image is, and were kept out of the 0–100 score to
  avoid double-counting the sharpness/edge signal they partially
  re-derive.


## GitHub Submission Checklist

- Keep `README.md` at the repository root.
- Keep `statement.md` at the repository root.
- Keep source modules inside `core/`, `analysis/`, `enhancement/`, `reports/`, and `config/`.
- Keep tests inside `tests/`.
- Keep generated runtime files inside `outputs/`; they are ignored by Git except for `outputs/.gitkeep`.
- Do not commit virtual environments, Python caches, or generated reports.
- Run the test suite before pushing:

```bash
python -m pytest tests/ -v
```

Expected result for the supplied project test suite: **21 passed**.

### Adaptive Enhancement Selection

The `--enhance` mode is self-checking. PixelPulse proposes corrections from the diagnosis, applies each candidate in sequence, re-runs the quality engine, and accepts an operation only when it produces a meaningful overall score improvement. This helps avoid blindly applying filters that improve one metric while harming the image overall.
