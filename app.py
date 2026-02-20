import streamlit as st
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from PIL import Image
import os
import io
import time

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FashionLens · AI Style Finder",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS — Dark editorial aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg:        #0a0a0a;
    --surface:   #111111;
    --surface2:  #1a1a1a;
    --border:    #2a2a2a;
    --accent:    #c8a96e;
    --accent2:   #e8c99e;
    --text:      #f0ece4;
    --muted:     #666;
    --danger:    #e05c5c;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.hero-logo {
    font-family: 'Playfair Display', serif;
    font-size: 3.8rem;
    font-weight: 900;
    letter-spacing: -1px;
    background: linear-gradient(135deg, var(--accent), var(--accent2), #fff8f0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.5rem;
}
.hero-rule {
    width: 48px;
    height: 2px;
    background: var(--accent);
    margin: 1.2rem auto 0;
    border: none;
}

/* ── Section labels ── */
.section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.8rem;
    display: block;
}

/* ── Upload zone ── */
.upload-zone {
    border: 1.5px dashed var(--border);
    border-radius: 12px;
    padding: 2.5rem 1.5rem;
    text-align: center;
    background: var(--surface2);
    transition: border-color 0.3s;
}
.upload-zone:hover { border-color: var(--accent); }

/* ── Query image container ── */
.query-frame {
    position: relative;
    border: 2px solid var(--accent);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 0 40px rgba(200,169,110,0.15);
}
.query-label {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    background: linear-gradient(transparent, rgba(0,0,0,0.85));
    color: var(--accent);
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    text-align: center;
    padding: 1rem 0 0.5rem;
}

/* ── Result cards ── */
.rec-card {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    background: var(--surface);
    transition: transform 0.25s, border-color 0.25s, box-shadow 0.25s;
    cursor: pointer;
}
.rec-card:hover {
    transform: translateY(-4px);
    border-color: var(--accent);
    box-shadow: 0 8px 32px rgba(200,169,110,0.12);
}
.rec-rank {
    font-family: 'Playfair Display', serif;
    font-size: 0.7rem;
    letter-spacing: 0.3em;
    color: var(--muted);
    text-align: center;
    padding: 0.5rem 0 0.3rem;
}
.rec-score {
    font-size: 0.75rem;
    color: var(--accent);
    text-align: center;
    padding: 0.2rem 0 0.6rem;
    font-weight: 500;
}

/* ── Stats bar ── */
.stats-bar {
    display: flex;
    gap: 2rem;
    padding: 1.2rem 1.5rem;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 2rem;
}
.stat-item { text-align: center; flex: 1; }
.stat-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
}
.stat-label {
    font-size: 0.68rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--muted);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    padding: 0.6rem 1.8rem !important;
    transition: opacity 0.2s, transform 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Sliders & selects ── */
[data-testid="stSlider"] > div > div > div {
    background: var(--accent) !important;
}
.stSelectbox label, .stSlider label, .stFileUploader label {
    color: var(--text) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Metric ── */
[data-testid="stMetric"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
}
[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    color: var(--accent) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface2);
    border: 1.5px dashed var(--border);
    border-radius: 12px;
    padding: 1rem;
}

/* ── Info / warning boxes ── */
.stAlert { border-radius: 10px !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
.stTabs [aria-selected="true"] {
    border-bottom-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Toast / spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS & PATHS  (edit these for your setup)
# ─────────────────────────────────────────────
if "IMAGE_DIR" not in st.session_state:
    st.session_state.IMAGE_DIR = "images"
if "PKL_FEATURES" not in st.session_state:
    st.session_state.PKL_FEATURES = "Images_features.pkl"
if "PKL_FILENAMES" not in st.session_state:
    st.session_state.PKL_FILENAMES = "filenames.pkl"

# ─────────────────────────────────────────────
#  CACHED MODEL & DATA LOADERS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    m = tf.keras.models.Sequential([base, GlobalMaxPool2D()])
    return m

@st.cache_resource(show_spinner=False)
def load_data(pkl_features_path, pkl_filenames_path):
    features  = pkl.load(open(pkl_features_path,  'rb'))
    filenames = pkl.load(open(pkl_filenames_path, 'rb'))
    return np.array(features), filenames

@st.cache_resource(show_spinner=False)
def load_knn(_features, n_neighbors):
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='brute', metric='euclidean')
    knn.fit(_features)
    return knn

# ─────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_features(img_input, model):
    """Accept PIL Image or file path, return normalised feature vector."""
    if isinstance(img_input, str):
        img = keras_image.load_img(img_input, target_size=(224, 224))
    else:
        img = img_input.convert("RGB").resize((224, 224))

    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    feat = model.predict(arr, verbose=0).flatten()
    return feat / norm(feat)

# ─────────────────────────────────────────────
#  SIMILARITY SCORE HELPER
# ─────────────────────────────────────────────
def dist_to_score(dist):
    return max(0, round((1 - dist / 2) * 100, 1))

# ─────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-logo">FashionLens</div>
    <div class="hero-sub">AI-Powered Visual Style Finder</div>
    <hr class="hero-rule">
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="section-label">⚙ Configuration</span>', unsafe_allow_html=True)

    n_results = st.slider("Number of Recommendations", 1, 10, 5)
    show_scores = st.toggle("Show Similarity Scores", value=True)
    show_filenames = st.toggle("Show Filenames", value=False)

    st.markdown("---")
    st.markdown('<span class="section-label">ℹ About</span>', unsafe_allow_html=True)
    st.caption(
        "FashionLens uses **ResNet50** to extract 2048-dim visual embeddings, "
        "then **K-Nearest Neighbours** (Euclidean) to retrieve the most visually "
        "similar items from your wardrobe catalog."
    )

# ─────────────────────────────────────────────
#  CHECK FILES EXIST
# ─────────────────────────────────────────────
files_ok = os.path.exists(st.session_state.PKL_FEATURES) and os.path.exists(st.session_state.PKL_FILENAMES)

if not files_ok:
    st.warning(
        f"⚠️  Pre-computed feature files not found at **{st.session_state.PKL_FEATURES}** / **{st.session_state.PKL_FILENAMES}**.  \n"
        "Please run the Colab notebook first to generate them, then place them in the same directory as `app.py`."
    )
    st.stop()

# ─────────────────────────────────────────────
#  LOAD RESOURCES
# ─────────────────────────────────────────────
with st.spinner("Loading ResNet50 model…"):
    model = load_model()

with st.spinner("Loading feature index…"):
    features, filenames = load_data(st.session_state.PKL_FEATURES, st.session_state.PKL_FILENAMES)
    knn_model = load_knn(features, n_results)

# ─────────────────────────────────────────────
#  STATS BAR
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-value">{len(filenames):,}</div>
        <div class="stat-label">Items Indexed</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">{features.shape[1]:,}</div>
        <div class="stat-label">Feature Dims</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">ResNet50</div>
        <div class="stat-label">Backbone</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">KNN</div>
        <div class="stat-label">Retrieval</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍  Find Similar Items", "🗂  Browse Catalog"])

# ══════════════════════════════════════════════
#  TAB 1 — RECOMMENDER
# ══════════════════════════════════════════════
with tab1:
    left_col, right_col = st.columns([1, 2], gap="large")

    # ── LEFT: Input ──────────────────────────
    with left_col:
        st.markdown('<span class="section-label">Upload or Select</span>', unsafe_allow_html=True)

        input_mode = st.radio(
            "Image source",
            ["Upload your image", "Pick from catalog"],
            horizontal=True,
            label_visibility="collapsed",
        )

        query_img = None
        query_label = ""

        if input_mode == "Upload your image":
            uploaded = st.file_uploader(
                "Drop a fashion image here",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
            )
            if uploaded:
                query_img   = Image.open(uploaded)
                query_label = uploaded.name

        else:  # Pick from catalog
            if len(filenames) > 0:
                sample_idx = st.number_input(
                    f"Image index (0 – {len(filenames)-1})",
                    min_value=0, max_value=len(filenames)-1,
                    value=0, step=1,
                )
                query_img   = Image.open(filenames[sample_idx])
                query_label = os.path.basename(filenames[sample_idx])
            else:
                st.error("No images found in catalog.")

        if query_img:
            st.markdown("---")
            st.markdown('<span class="section-label">Query Image</span>', unsafe_allow_html=True)
            st.image(query_img, use_column_width=True, caption=query_label)

            if st.button("✦  Find Similar Styles"):
                st.session_state.find_button_clicked = True
            else:
                st.session_state.find_button_clicked = False

    # ── RIGHT: Results ───────────────────────
    with right_col:
        st.markdown('<span class="section-label">Similar Items</span>', unsafe_allow_html=True)

        if query_img and st.session_state.get('find_button_clicked', False):
            with st.spinner("Extracting visual features…"):
                t0       = time.time()
                q_feat   = extract_features(query_img, model)
                distances, indices = knn_model.kneighbors([q_feat])
                elapsed  = time.time() - t0

            # Skip index 0 (query itself if from catalog) or keep all if uploaded
            rec_indices   = indices[0][1:n_results+1]
            rec_distances = distances[0][1:n_results+1]

            st.caption(f"⚡ Retrieved in {elapsed:.2f}s")

            # ── Grid of results
            cols_per_row = min(n_results, 5)
            rows = (n_results + cols_per_row - 1) // cols_per_row

            result_ptr = 0
            for row in range(rows):
                grid_cols = st.columns(cols_per_row)
                for col_i, col in enumerate(grid_cols):
                    if result_ptr >= len(rec_indices):
                        break
                    idx  = rec_indices[result_ptr]
                    dist = rec_distances[result_ptr]
                    score = dist_to_score(dist)
                    fpath = filenames[idx]

                    with col:
                        try:
                            rec_img = Image.open(fpath)
                            st.image(rec_img, use_column_width=True)

                            meta = f"**#{result_ptr + 1}**"
                            if show_scores:
                                meta += f"  ·  `{score}%`"
                            st.markdown(meta)

                            if show_filenames:
                                st.caption(os.path.basename(fpath))
                        except Exception:
                            st.warning(f"Could not load {fpath}")

                    result_ptr += 1

        elif query_img:
            st.info("👈  Click **Find Similar Styles** to run the recommendation.")
        else:
            st.markdown("""
            <div style='text-align:center; color:#444; padding: 5rem 0;'>
                <div style='font-size:3rem;'>👗</div>
                <div style='font-family: Playfair Display, serif; font-size:1.2rem; color:#555; margin-top:1rem;'>
                    Upload an image to discover similar styles
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 2 — BROWSE CATALOG
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<span class="section-label">Catalog Browser</span>', unsafe_allow_html=True)

    page_size = st.select_slider("Items per page", options=[12, 24, 48, 96], value=24)
    total_pages = max(1, (len(filenames) + page_size - 1) // page_size)
    page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1) - 1

    start = page_num * page_size
    end   = min(start + page_size, len(filenames))
    page_files = filenames[start:end]

    st.caption(f"Showing items {start+1}–{end} of {len(filenames):,}")

    grid_cols = st.columns(6)
    for i, fpath in enumerate(page_files):
        with grid_cols[i % 6]:
            try:
                img = Image.open(fpath)
                st.image(img, use_column_width=True)
                if show_filenames:
                    st.caption(os.path.basename(fpath))
            except Exception:
                st.warning("?")
