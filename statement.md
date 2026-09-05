# Problem Statement — PixelPulse

## 1. Background

Anyone who has taken or received digital photographs has run into images
that look "off" — blurry, washed out, too dark, too noisy — without an
easy way to say *why*, or *how much*, or *what would fix it*. Most
consumer tools either apply a one-size-fits-all "auto-enhance" filter
regardless of what is actually wrong with the picture, or require a human
to eyeball the image and guess.

## 2. Problem

There is no lightweight, transparent, command-line tool that:

1. Measures multiple independent dimensions of image quality
   (sharpness, contrast, noise, exposure, edge structure, texture) using
   well-understood, explainable classical image-processing techniques,
2. Combines those measurements into a single, justified quality score
   rather than a black-box "good/bad" label,
3. States *specifically* what is wrong with the image and *why*, backed
   by the numeric evidence behind the diagnosis, and
4. Recommends — and, if requested, applies — only the corrective filters
   that are actually warranted, instead of blindly running every
   enhancement on every image.

## 3. Objective

Build **PixelPulse**, a command-line Intelligent Image Health & Quality
Analyzer that takes an image as input and outputs:

- A quality score (0–100) and category (POOR / WEAK / ACCEPTABLE / GOOD /
  EXCELLENT),
- A breakdown of the underlying sub-scores (sharpness, contrast, noise,
  illumination, edge preservation, texture),
- A list of detected problems with supporting numeric evidence,
- A targeted enhancement recommendation, optionally applied and
  re-scored to demonstrate measurable improvement.

## 4. Scope

**In scope:**
- Single-image analysis, diagnosis, and optional enhancement via CLI.
- Classical image processing only: histograms, Laplacian/Canny/CLAHE,
  Non-Local Means denoising, unsharp masking, HOG, SIFT.
- Before/after comparison mode to quantitatively evaluate an enhancement.
- Text and JSON report generation, plus histogram and edge-map plots.

**Out of scope:**
- Deep-learning-based quality assessment or enhancement models.
- Batch/video processing.
- A graphical user interface (the project is explicitly CLI-only per the
  evaluation requirements).

## 5. Functional Requirements

| ID | Requirement |
|----|-------------|
| FR1 | The system shall load and validate an image file, rejecting missing, empty, corrupt, unsupported-format, or too-small files with a clear error message. |
| FR2 | The system shall compute brightness, contrast, noise, blur/sharpness, edge density, and texture-energy metrics for a given image. |
| FR3 | The system shall combine the above metrics into a single weighted 0–100 quality score and a quality category label. |
| FR4 | The system shall identify specific quality problems (e.g. blur, noise, low contrast, over/under-exposure) with supporting numeric evidence. |
| FR5 | The system shall recommend specific corrective enhancement(s) for each detected problem. |
| FR6 | The system shall, on request, apply only the recommended enhancement(s) and save the result. |
| FR7 | The system shall re-analyze an enhanced image and report the before/after score delta. |
| FR8 | The system shall generate a human-readable text report and a machine-readable JSON report. |
| FR9 | The system shall optionally save a histogram plot and an edge-map image as visual evidence. |
| FR10 | All functionality shall be accessible from the command line without any GUI dependency. |

## 6. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR1 | The system must run entirely offline (no network calls, no model downloads). |
| NFR2 | Analysis of a single typical photograph (≤ 1600px on the longer side after internal resizing) must complete in a few seconds on a standard laptop CPU. |
| NFR3 | All scoring weights and thresholds must be centralized and documented (`config/settings.py`) rather than hard-coded inline, so they can be justified and adjusted. |
| NFR4 | The codebase must be modular (separate loading, analysis, scoring, recommendation, and reporting concerns) so each part is independently testable. |
| NFR5 | Core logic must be covered by automated tests (`tests/`), runnable via `pytest`. |

## 7. Tech Stack

- **Language:** Python 3
- **Libraries:** OpenCV (`opencv-python-headless`), NumPy, scikit-image
  (HOG), Matplotlib (report plots), pytest (testing)
- **Interface:** Command line (`argparse`)

## 8. Deliverables

- Public GitHub repository containing the full source code, tests, and
  sample images.
- `README.md` with complete setup and usage instructions.
- This `statement.md`.
- A structured project report (submitted separately per the course
  platform's required format), covering architecture, design diagrams,
  implementation details, experimental results, testing, and reflections.


### Enhancement Validation

The enhancement stage is quality-aware rather than purely rule-based. Recommended operations are treated as candidates, and each candidate is re-evaluated with the same quality engine before it is accepted. An operation is retained only when it produces a meaningful improvement in the overall score.
