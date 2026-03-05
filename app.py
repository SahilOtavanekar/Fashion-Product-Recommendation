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
import time

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FashionLens · AI Style Finder (Lite)",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0a0a0a;
    --surface:   #111111;
    --surface2:  #1a1a1a;
    --border:    #2a2a2a;
    --accent:    #c8a96e;
    --accent2:   #e8c99e;
    --text:      #f0ece4;
    --muted:     #666;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }

[data-testid="stSidebar"] {
    background-color: var(--surface);
    border-right: 1px solid var(--border);
}

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

.section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.8rem;
    display: block;
}

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

.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
DATA_DIR = "data"
IMAGE_DIR = "images"
FEATURES_PATH = os.path.join(DATA_DIR, "features.pkl")
FILENAMES_PATH = os.path.join(DATA_DIR, "filenames.pkl")

# ─────────────────────────────────────────────
#  LOADERS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    model = tf.keras.models.Sequential([base, GlobalMaxPool2D()])
    return model

@st.cache_resource(show_spinner=False)
def load_data():
    if not os.path.exists(FEATURES_PATH) or not os.path.exists(FILENAMES_PATH):
        return None, None
    with open(FEATURES_PATH, 'rb') as f:
        features = pkl.load(f)
    with open(FILENAMES_PATH, 'rb') as f:
        filenames = pkl.load(f)
    return np.array(features), filenames

@st.cache_resource(show_spinner=False)
def load_knn(_features, n_neighbors):
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='brute', metric='euclidean')
    knn.fit(_features)
    return knn

def extract_features(img_input, model):
    if isinstance(img_input, str):
        img = keras_image.load_img(img_input, target_size=(224, 224))
    else:
        img = img_input.convert("RGB").resize((224, 224))
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    feat = model.predict(arr, verbose=0).flatten()
    return feat / norm(feat)

def dist_to_score(dist):
    return max(0, round((1 - dist / 2) * 100, 1))

# ─────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-logo">FashionLens Lite</div>
    <div class="hero-sub">AI-Powered Visual Style Finder</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<span class="section-label">⚙ Configuration</span>', unsafe_allow_html=True)
    n_results = st.slider("Recommendations", 1, 10, 5)
    show_scores = st.toggle("Show Similarity %", value=True)
    
    st.markdown("---")
    st.caption("Demo version with 2,000 items. Uses ResNet50 + KNN.")

# Load Resources
model = load_model()
features, filenames = load_data()

if features is None:
    st.error("Catalog data missing. Please check the 'data' folder.")
    st.stop()

knn_model = load_knn(features, n_results)

st.markdown(f"""
<div class="stats-bar">
    <div class="stat-item"><div class="stat-value">{len(filenames):,}</div><div class="stat-label">Items</div></div>
    <div class="stat-item"><div class="stat-value">ResNet50</div><div class="stat-label">Model</div></div>
    <div class="stat-item"><div class="stat-value">Lite</div><div class="stat-label">Edition</div></div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Find", "🗂 Browse"])

with tab1:
    uploaded = st.file_uploader("Upload fashion image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        query_img = Image.open(uploaded)
        st.image(query_img, width=200, caption="Your Upload")
        
        if st.button("Find Similar Styles"):
            with st.spinner("Analyzing style..."):
                q_feat = extract_features(query_img, model)
                distances, indices = knn_model.kneighbors([q_feat])
            
            rec_indices = indices[0][1:]
            rec_distances = distances[0][1:]
            
            cols = st.columns(len(rec_indices))
            for i, col in enumerate(cols):
                idx = rec_indices[i]
                dist = rec_distances[i]
                fpath = filenames[idx]
                with col:
                    if os.path.exists(fpath):
                        st.image(Image.open(fpath), use_column_width=True)
                        if show_scores:
                            st.caption(f"Match: {dist_to_score(dist)}%")
                    else:
                        st.warning("Missing")

with tab2:
    page = st.number_input("Page", min_value=1, value=1)
    page_size = 18
    start = (page-1)*page_size
    end = start + page_size
    files = filenames[start:end]
    
    grid = st.columns(6)
    for i, fpath in enumerate(files):
        with grid[i % 6]:
            if os.path.exists(fpath):
                st.image(Image.open(fpath), use_column_width=True)
