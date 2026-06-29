<div align="center">

<img src="./assets/banner.svg" alt="CLIP Similar & Duplicate Image Finder" width="100%"/>

![Last Commit](https://img.shields.io/github/last-commit/zibadehghani/clip-similar-image-finder?color=2563EB&label=last%20update&style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-2563EB?style=flat-square&logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/UI-Gradio-FF7C00?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-0F2A5C?style=flat-square)

### 🚀 [Try the live demo on Hugging Face Spaces](https://huggingface.co/spaces/ZibaDehghani/clip-similar-image-finder)

</div>

# CLIP Similar & Duplicate Image Finder

A small computer-vision tool that uses OpenAI's **CLIP** model to find visually/semantically similar images in a personal photo collection, with automatic **exact-duplicate** flagging.

Upload an image through the web app and get back the most similar images from your dataset, ranked by cosine similarity — adjustable in real time with a similarity-threshold slider.

## How it works

```
images folder → CLIP image encoder → 512-dim embeddings
                                          │
query image  → CLIP image encoder ──► cosine similarity ──► ranked results
                                          │
                              similarity ≥ 0.98 → flagged as "Exact Duplicate"
```

## Features

- 🔍 Semantic search with CLIP — matches by visual/conceptual similarity, not pixels
- 🎛️ Adjustable similarity threshold (slider), no fixed result count
- 🚩 Automatic exact-duplicate detection, separate from general matches
- 📁 Optional custom dataset — upload your own images in the app and search against them instead of the bundled demo dataset
- 🖼️ Clean Gradio UI themed to match this project

## Key finding

CLIP tends to match images by overall **composition, lighting, and color palette** rather than exact object category — e.g. close-up animal portraits with blurred backgrounds scored as similar to each other regardless of species. The meaningful similarity range is also dataset-dependent, so the threshold should be calibrated per dataset rather than fixed.

## Run locally

```bash
pip install -r requirements.txt
```

Add your image collection to a folder named `images/` next to `app.py`, then:

```bash
python app.py
```

Then open the local URL shown in the terminal, upload a query image, and adjust the threshold slider.

## Tech stack

CLIP (`openai/clip-vit-base-patch32`) · Hugging Face `transformers` · PyTorch · Gradio

## Project structure

```
.
├── CLIP Similar & Duplicate Image Finder.ipynb   # full notebook, step by step
├── app.py                                        # standalone Gradio app (reads from ./images)
├── requirements.txt
└── assets/banner.svg
```

---

<div align="center">

Built by **Ziba Dehghani** — [github.com/zibadehghani](https://github.com/zibadehghani)

</div>
