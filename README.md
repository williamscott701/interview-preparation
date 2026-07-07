# Interview Preparation — ML / DL & Computer Vision Notes

A curated knowledge base for **Machine Learning, Deep Learning, and Computer Vision** interviews (with an Applied Scientist / Amazon focus). The notes are written in Markdown and rendered into a polished, self-contained website.

### 🌐 Live site

**https://williamscott701.github.io/interview-preparation/**

The landing page opens the ML/DL notes; use the **Computer Vision ↗** button in the top bar to switch between the two documents.

---

## 📚 What's inside

### 1. ML / DL Interview Notes — [`ML_DL_Notes.html`](ML_DL_Notes.html) · source [`ML DL/ML_DL_Notes.md`](ML%20DL/ML_DL_Notes.md)

14 sections, 70+ subsections, covering the breadth of an Applied Scientist interview:

| # | Section | # | Section |
|---|---------|---|---------|
| 1 | Mathematical Foundations | 8 | Optimization |
| 2 | Statistical Inference | 9 | Neural Network Fundamentals |
| 3 | Core ML Concepts | 10 | Dimensionality Reduction |
| 4 | Feature Engineering & Data | 11 | Sequential Models |
| 5 | Classical ML Models | 12 | NLP & Transformers |
| 6 | Ensemble Methods | 13 | Advanced Topics |
| 7 | Regularization & Loss Functions | 14 | Amazon Applied Scientist Interview Topics |

Plus a large **References** section with curated reading, grouped by topic.

### 2. Computer Vision Interview Notes — [`CV_Notes.html`](CV_Notes.html) · source [`CV/CV_Notes.md`](CV/CV_Notes.md)

| # | Section | Highlights |
|---|---------|-----------|
| 1 | Generative Models | Autoencoders, VAEs, GANs — architecture, objectives, trade-offs |
| 2 | CNN Fundamentals | Conv layers, pooling, activations, architectures, receptive field, kernels |
| 3 | CNN Advanced Concepts | Padding, stride, dilated convs, BatchNorm vs LayerNorm, ResNets, transfer learning |

---

## ✨ Reading experience

The HTML pages are **single self-contained files** (inline CSS/JS — only [Temml](https://temml.org/) and the Google Fonts web fonts load from a CDN) and include:

- 🎨 **Per-section accent colors** and a collapsible **sidebar** with live **scroll-spy**
- 🌙 **Dark mode** toggle and a **reading-progress** bar
- ∑ **LaTeX math** via Temml, rendered client-side to native (searchable/selectable) MathML — with multi-line equations aligned on `=`
- 🖼️ **Image controls** — hover any figure for its filename and a size control (S / M / L / Full + slider); the chosen size is **remembered per image**
- 🔗 GitHub-style heading anchors and a cross-link between the two documents

---

## 🗂️ Repository layout

```
.
├── index.html            # Redirects to ML_DL_Notes.html (GitHub Pages entry point)
├── ML_DL_Notes.html      # Generated ML/DL site  (images → ML%20DL/images/…)
├── CV_Notes.html         # Generated Computer Vision site  (images → CV/images/…)
├── .nojekyll             # Serve folders/paths verbatim on GitHub Pages
├── CHANGELOG.md          # Version history shown in the top bar of both pages
│
├── ML DL/
│   ├── ML_DL_Notes.md    # Source notes (edit this)
│   ├── generate_html.py  # Markdown → HTML generator
│   ├── images/           # Figures
│   └── raw/              # Original Google-Doc export (archive)
│
└── CV/
    ├── CV_Notes.md
    ├── generate_html.py
    ├── images/
    └── raw/
```

> The **Markdown files are the source of truth** — the `.html` files are generated artifacts.

---

## 🔧 Building the site

The generators turn each Markdown file into its styled HTML page at the repo root.

**Requirements:** Python 3 and [`markdown-it-py`](https://github.com/executablebooks/markdown-it-py):

```bash
pip install markdown-it-py
```

**Regenerate** (run from the repo root):

```bash
python3 "ML DL/generate_html.py"   # → ML_DL_Notes.html
python3 "CV/generate_html.py"      # → CV_Notes.html
```

### How it works

`generate_html.py` runs a small pipeline:

1. **Protect math** — `$…$` / `$$…$$` are stashed so Markdown can't mangle them, and consecutive `=` lines are merged into an aligned block.
2. **Render** with `markdown-it-py` (CommonMark + GFM tables + raw HTML) so output matches a Markdown preview exactly.
3. **Post-process** — restore math, add GitHub-style heading IDs, wrap each `##` section in a colored card (the title block and Table of Contents are dropped from the page since the top bar and sidebar replace them), and rewrite image paths for the repo-root layout.
4. **Assemble** the final document with inline CSS/JS, the Temml math renderer, the sidebar, and the image/size controls.

The script installs `markdown-it-py` automatically via `pip` on first run if it isn't already present. Math (Temml) and the Inter/JetBrains Mono web fonts are loaded from CDNs at page view time, so viewing the HTML requires an internet connection; everything else (layout, dark mode, image controls) works offline.

---

## 📝 Notes

- Content is compiled from personal interview-preparation materials and a range of public resources (linked in the References section of the ML/DL notes).
- The `raw/` folders keep the original Google-Doc exports for reference.
- See [`CHANGELOG.md`](CHANGELOG.md) for the version history (the version number shown in the top bar of both pages is bumped on every change).
