"""
CLIP Similar & Duplicate Image Finder
A Gradio app that uses CLIP embeddings to find visually/semantically similar
images in a personal dataset, with automatic exact-duplicate flagging.

This mirrors the logic in clip_similar_image_finder.ipynb, adapted to:
  1) read a bundled demo dataset from an "images" folder, and
  2) optionally let the user upload and use their own dataset instead.
"""

import os
import glob

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import gradio as gr

# ---------------------------------------------------------------------------
# Setup: load model once at startup
# ---------------------------------------------------------------------------

# Try a couple of common folder name spellings for the bundled demo dataset
DEFAULT_FOLDER_CANDIDATES = ["images", "Images"]

# Base CLIP model: good accuracy/speed tradeoff, runs fine on CPU
MODEL_NAME = "openai/clip-vit-base-patch32"

# Images scoring this high are treated as exact duplicates rather than just
# "similar", since even closely related images rarely reach this score with CLIP
DUPLICATE_THRESHOLD = 0.98

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)


def embed_images(pil_images):
    """Encode a list of PIL images into CLIP embeddings (N x 512)."""
    inputs = processor(images=pil_images, return_tensors="pt")
    with torch.no_grad():  # inference only, no need to track gradients
        output = model.get_image_features(**inputs)
        # Different transformers versions expose the embedding under different names
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        elif hasattr(output, "image_embeds"):
            return output.image_embeds
        else:
            return output


def load_default_dataset():
    """Load and embed the bundled demo dataset, trying a couple of folder name spellings."""
    for folder in DEFAULT_FOLDER_CANDIDATES:
        paths = glob.glob(os.path.join(folder, "*.*"))
        if paths:
            images, valid_paths = [], []
            for path in paths:
                try:
                    images.append(Image.open(path).convert("RGB"))
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Failed to open {path}: {e}")
            if images:
                print(f"Loaded {len(images)} demo dataset images from '{folder}'.")
                return embed_images(images), valid_paths
    print("No demo dataset folder found; default dataset is empty.")
    return None, []


default_embeddings, default_paths = load_default_dataset()


# ---------------------------------------------------------------------------
# Custom dataset: build embeddings from user-uploaded files
# ---------------------------------------------------------------------------

def build_custom_dataset(files):
    """
    Encode user-uploaded images into a custom dataset (embeddings + paths),
    stored in gr.State so it can be reused across searches without recomputing.
    """
    if not files:
        return None, None, "No files uploaded yet."

    images, valid_paths = [], []
    for f in files:
        path = f.name if hasattr(f, "name") else f
        try:
            images.append(Image.open(path).convert("RGB"))
            valid_paths.append(path)
        except Exception as e:
            print(f"Failed to open {path}: {e}")

    if not images:
        return None, None, "Could not read any of the uploaded files as images."

    embeddings = embed_images(images)
    status = f"Custom dataset ready: {len(images)} image(s) indexed. Now click 'Find Similar Images' above."
    return embeddings, valid_paths, status


def clear_custom_dataset():
    """Discard the custom dataset and fall back to the bundled demo dataset."""
    return None, None, "Custom dataset cleared — using the bundled demo dataset again."


# ---------------------------------------------------------------------------
# Core search function
# ---------------------------------------------------------------------------

def gradio_find_similar(query_image, threshold, custom_embeddings, custom_paths):
    """
    Find images similar to an uploaded query image using CLIP embeddings.
    Uses the user's custom dataset (if one has been built) instead of the
    bundled demo dataset. Returns (image, caption) pairs for the Gallery,
    flagging exact duplicates (similarity >= DUPLICATE_THRESHOLD) separately.
    """
    if custom_embeddings is not None and custom_paths:
        embeddings, valid_paths, source = custom_embeddings, custom_paths, "your uploaded dataset"
    else:
        embeddings, valid_paths, source = default_embeddings, default_paths, "the demo dataset"

    if embeddings is None or not valid_paths:
        return [], "No dataset available. Upload images in 'Use your own dataset' below, or add a demo dataset."

    query_input = processor(images=[query_image], return_tensors="pt")
    with torch.no_grad():
        query_output = model.get_image_features(**query_input)
        if hasattr(query_output, "pooler_output"):
            query_embedding = query_output.pooler_output
        elif hasattr(query_output, "image_embeds"):
            query_embedding = query_output.image_embeds
        else:
            query_embedding = query_output

    scores = F.cosine_similarity(query_embedding, embeddings, dim=1)
    scores_sorted = torch.argsort(scores, descending=True)

    result_images, result_captions = [], []
    duplicate_count = 0

    for idx in scores_sorted:
        idx = idx.item()
        score = scores[idx].item()
        if score < threshold:
            break  # remaining scores are even lower, no need to keep checking
        img = Image.open(valid_paths[idx]).convert("RGB")
        if score >= DUPLICATE_THRESHOLD:
            caption = f"Exact Duplicate ({score:.2f})"
            duplicate_count += 1
        else:
            caption = f"Similar ({score:.2f})"
        result_images.append(img)
        result_captions.append(caption)
        if len(result_images) == 10:
            break

    if not result_images:
        return [], f"No similar images found in {source}. Try lowering the threshold."

    note = f"Searched {source} — found {len(result_images)} match(es), {duplicate_count} exact duplicate(s)."
    return list(zip(result_images, result_captions)), note


# ---------------------------------------------------------------------------
# UI: theme and layout matched to the project banner (blue / white)
# ---------------------------------------------------------------------------

custom_theme = gr.themes.Soft(primary_hue="blue", secondary_hue="blue", neutral_hue="slate").set(
    body_background_fill="#F4F8FF",
    block_background_fill="#FFFFFF",
    block_border_color="#DCE8FF",
    button_primary_background_fill="#1E54B7",
    button_primary_background_fill_hover="#3B82F6",
    button_primary_text_color="#FFFFFF",
)

custom_css = """
#title-block {text-align:center; padding: 6px 0 0 0;}
#title-block h1 {color:#0F2A5C; font-weight:800;}
#subtitle {text-align:center; color:#0F2A5C; margin-top:-8px; font-weight:600;}
.gradio-container {max-width: 1100px !important; margin: auto;}

#dataset-note {
    color:#0F2A5C;
    font-weight:600;
    background:#EEF4FF;
    padding:10px 14px;
    border-radius:8px;
    margin-bottom:8px;
}

/* Similarity threshold slider tooltip */
#threshold-slider { position: relative; }
#threshold-slider::after {
    content: "Lower = looser matches (more results) · Higher = stricter matches (fewer, closer results)";
    display: none;
    position: absolute; top: -34px; left: 0;
    background: #0F2A5C; color: #fff; font-size: 12px; padding: 6px 10px;
    border-radius: 6px; white-space: nowrap; z-index: 10;
}
#threshold-slider:hover::after { display: block; }

#build-dataset-btn, #clear-dataset-btn { position: relative; }

#build-dataset-btn::after {
    content: "Encode your uploaded images and search against them";
    display: none;
    position: absolute; top: 108%; left: 50%; transform: translateX(-50%);
    background: #0F2A5C; color: #fff; font-size: 12px; padding: 6px 10px;
    border-radius: 6px; white-space: nowrap; z-index: 10;
}
#build-dataset-btn:hover::after { display: block; }

#clear-dataset-btn::after {
    content: "Discard your custom dataset and use the demo dataset again";
    display: none;
    position: absolute; top: 108%; left: 50%; transform: translateX(-50%);
    background: #0F2A5C; color: #fff; font-size: 12px; padding: 6px 10px;
    border-radius: 6px; white-space: nowrap; z-index: 10;
}
#clear-dataset-btn:hover::after { display: block; }

/* Accordion label + arrow styling */
#dataset-accordion .label-wrap span {
    color:#0F2A5C !important;
    font-weight:700 !important;
    font-size: 15px !important;
}
#dataset-accordion .label-wrap svg {
    width: 20px !important;
    height: 20px !important;
    color: #0F2A5C !important;
    fill: #0F2A5C !important;
    stroke: #0F2A5C !important;
}
"""

with gr.Blocks(theme=custom_theme, css=custom_css) as demo:
    # Holds the user's custom dataset for the duration of their session
    custom_embeddings_state = gr.State(None)
    custom_paths_state = gr.State(None)

    with gr.Column(elem_id="title-block"):
        gr.Markdown("# 🔎 CLIP Similar & Duplicate Image Finder")
        gr.Markdown(
            "Upload an image to find visually/semantically similar images, "
            "with automatic flagging of exact duplicates.",
            elem_id="subtitle",
        )

    with gr.Row():
        with gr.Column():
            query_input = gr.Image(type="pil", label="Upload a query image")
            threshold_input = gr.Slider(
                minimum=0.5, maximum=0.95, value=0.7, step=0.01,
                label="Similarity threshold",
                elem_id="threshold-slider",
            )
            submit_btn = gr.Button("Find Similar Images", variant="primary")
        with gr.Column():
            gallery_output = gr.Gallery(label="Most similar images", columns=3, height=400)
            summary_output = gr.Textbox(label="Result summary")

    with gr.Accordion("📁 Use your own dataset (optional)", open=False, elem_id="dataset-accordion"):
        gr.Markdown(
            "By default, searches run against the bundled demo dataset. "
            "Upload your own images below to search against them instead "
            "for the rest of this session.",
            elem_id="dataset-note",
        )
        custom_files_input = gr.File(
            label="Upload your own images",
            file_count="multiple",
            file_types=["image"],
        )
        with gr.Row():
            build_dataset_btn = gr.Button("Build my dataset", variant="primary", elem_id="build-dataset-btn")
            clear_dataset_btn = gr.Button("Use demo dataset instead", variant="primary", elem_id="clear-dataset-btn")
        dataset_status = gr.Textbox(label="Dataset status", interactive=False)

    submit_btn.click(
        fn=gradio_find_similar,
        inputs=[query_input, threshold_input, custom_embeddings_state, custom_paths_state],
        outputs=[gallery_output, summary_output],
    )

    build_dataset_btn.click(
        fn=build_custom_dataset,
        inputs=[custom_files_input],
        outputs=[custom_embeddings_state, custom_paths_state, dataset_status],
    )

    clear_dataset_btn.click(
        fn=clear_custom_dataset,
        inputs=[],
        outputs=[custom_embeddings_state, custom_paths_state, dataset_status],
    )

if __name__ == "__main__":
    demo.launch()
