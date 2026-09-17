# PixelPulse — Intelligent Image Health & Quality Analyzer

PixelPulse is a command-line image quality assessment and enhancement system built using classical computer vision techniques. It analyzes an image, produces an explainable quality score from 0–100, identifies measurable quality problems, recommends appropriate enhancements, and can validate whether those enhancements actually improve the image quality.

The system is designed to operate offline on a CPU and does not use deep learning or pretrained models.

---

## Key Features

- Automated image loading and validation
- Brightness and illumination analysis
- Contrast analysis
- Blur and sharpness detection
- Noise estimation
- Edge preservation analysis
- Texture analysis
- HOG and SIFT diagnostic feature extraction
- Explainable 0–100 image quality score
- Quality categories:
  - POOR
  - WEAK
  - ACCEPTABLE
  - GOOD
  - EXCELLENT
- Numeric evidence for detected issues
- Automatic enhancement recommendations
- Adaptive enhancement validation
- Denoising
- Unsharp-mask sharpening
- CLAHE-based contrast enhancement
- Before/after score comparison
- Text and JSON reports
- Histogram and edge-map generation
- Automated unit testing with pytest
- Fully offline CLI execution

---

## Technology Stack

- Python
- OpenCV
- NumPy
- scikit-image
- Matplotlib
- pytest

No deep-learning framework or pretrained neural network is required.

---

## Project Structure

```text
PixelPulse/
├── analysis/
│   ├── blur_analysis.py
│   ├── edge_analysis.py
│   ├── feature_analysis.py
│   ├── histogram_analysis.py
│   ├── noise_analysis.py
│   └── texture_analysis.py
│
├── config/
│   └── settings.py
│
├── core/
│   ├── image_loader.py
│   ├── preprocessing.py
│   ├── quality_engine.py
│   └── recommendation_engine.py
│
├── data/
│   └── sample_images/
│       ├── degraded_sample.jpg
│       └── sharp_sample.jpg
│
├── enhancement/
│   ├── contrast.py
│   ├── denoise.py
│   └── sharpen.py
│
├── outputs/
│   └── .gitkeep
│
├── reports/
│   └── report_generator.py
│
├── tests/
│   ├── test_analysis.py
│   ├── test_loader.py
│   └── test_quality.py
│
├── main.py
├── requirements.txt
├── statement.md
├── .gitignore
├── LICENSE
└── README.md
---

## Repository

GitHub:

https://github.com/AryaAbhyankar0707/PixelPulse
