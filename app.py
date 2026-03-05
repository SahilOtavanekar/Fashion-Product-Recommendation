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
    page_title="FashionLens · Deep Learning Style Finder (Lite)",

    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:          #ffffff;
    --surface:     #f8fafc;
    --surface-alt: #f1f5f9;
    --accent:      #6366f1; /* Modern Indigo */
    --accent-glow: rgba(99, 102, 241, 0.1);
    --text-main:   #0f172a; /* Deep Slate */
    --text-muted:  #64748b;
    --border:      #e2e8f0;
    --success:     #10b981;
}



/* Base resets */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text-main);
}

.stApp { background-color: var(--bg); }
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 3rem 4rem; max-width: 1400px; }

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] .stMarkdown p {
    color: var(--text-muted) !important;
    font-size: 0.9rem;
}

/* Sidebar Toggle Contrast */
button[data-testid="stSidebarCollapse"] {
    color: var(--text-main) !important;
}


/* Headers */
.dash-header {
    margin-bottom: 2.5rem;
}

.dash-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 0.2rem;
    letter-spacing: -0.02em;
}


.dash-subtitle {
    color: var(--text-muted);
    font-size: 1.1rem;
}

/* Stats Container */
.stats-grid {
    display: flex;
    gap: 1.5rem;
    margin-bottom: 3rem;
}

.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    flex: 1;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.stat-card:hover {
    border-color: var(--accent);
}

.stat-val {
    font-family: 'Outfit', sans-serif;
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--accent);
    line-height: 1;
}

.stat-lab {
    color: var(--text-muted);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.5rem;
}

/* Feature Cards & Results */
.content-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 2rem;
}

.item-card {
    position: relative;
    background: var(--surface-alt);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.5rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.item-card:hover {
    transform: translateY(-5px);
    border-color: var(--accent);
    box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3), 0 0 15px var(--accent-glow);
}

.match-badge {
    position: absolute;
    top: 10px;
    right: 10px;
    background: rgba(15, 17, 21, 0.8);
    backdrop-filter: blur(4px);
    color: var(--success);
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.3rem 0.6rem;
    border-radius: 6px;
    border: 1px solid var(--success);
}

/* Buttons */
.stButton > button {
    width: 100% !important;
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 10px 15px -3px var(--accent-glow) !important;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    background-color: transparent !important;
    border-bottom: 1px solid var(--border) !important;
}

.stTabs [data-baseweb="tab"] {
    height: 45px !important;
    background-color: transparent !important;
    border-radius: 0 !important;
    color: var(--text-muted) !important;
}

.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 3px solid var(--accent) !important;
}

/* Custom Scrollbar */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #94a3b8; border-radius: 10px; border: 2px solid var(--bg); }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* Sidebar specific scrollbar */
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
    background: #64748b;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
DATA_DIR = "data"
IMAGE_DIR = "assets"
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
#  APP STATE
# ─────────────────────────────────────────────
if 'selected_img' not in st.session_state:
    st.session_state.selected_img = None

# ─────────────────────────────────────────────
#  UI DASHBOARD
# ─────────────────────────────────────────────
# Header
st.markdown("""
<div class="dash-header">
    <div class="dash-title">FashionLens <span style="color:var(--accent);">Deep Learning</span> Suite</div>
    <div class="dash-subtitle">Enterprise-Ready Visual Search Interface • Powered by TensorFlow</div>
</div>
""", unsafe_allow_html=True)


# Loading data once to avoid repeated calls
model = load_model()
features, filenames = load_data()

if features is None:
    st.error("Catalog data missing. Please check the 'data' folder.")
    st.stop()

# Dashboard Stats Grid
st.markdown(f"""
<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-val">🏷️ {len(filenames):,}</div>
        <div class="stat-lab">Catalog Volume</div>
    </div>
    <div class="stat-card">
        <div class="stat-val">🧠 ResNet50</div>
        <div class="stat-lab">Deep Analysis Layer</div>
    </div>
    <div class="stat-card">
        <div class="stat-val">⚡ Latency Lite</div>
        <div class="stat-lab">Search Optimized</div>
    </div>
</div>
""", unsafe_allow_html=True)


# Main Application Tabs
tab1, tab2 = st.tabs(["⚡ Search by Image", "� Catalog Browser"])

with st.sidebar:
    st.title("⚙️ Engine Control")
    n_results = st.slider("Max Recommendations", 1, 12, 6)
    show_scores = st.toggle("Show Match Confidence", value=True)
    st.markdown("---")
    st.markdown("### Development Info")
    st.info("System uses ResNet50 for feature embedding and Brute-force KNN with Euclidean distance for finding similar vector profiles.")

knn_model = load_knn(features, n_results)

with tab1:
    with st.container():
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1], gap="large")
        
        with col1:
            st.subheader("Current Intelligence Query")
            uploaded = st.file_uploader("Drop an image here to find matches", type=["jpg", "jpeg", "png", "webp"])
        
        with col2:
            if st.session_state.selected_img:
                st.write("**Selection from Catalog**")
                st.image(st.session_state.selected_img, use_container_width=True)
                if st.button("✕ Clear Selection", key="rst_btn"):
                    st.session_state.selected_img = None
                    st.rerun()
            else:
                st.info("No selection active")
        
        st.markdown('</div>', unsafe_allow_html=True)

    query_input = uploaded if uploaded else st.session_state.selected_img
    
    if query_input:
        img_display = Image.open(query_input) if uploaded else Image.open(st.session_state.selected_img)
        
        # Small query preview
        st.write("---")
        c1, c2, c3 = st.columns([2, 1, 2])
        with c2:
            st.markdown('<div class="item-card" style="margin-bottom:20px;">', unsafe_allow_html=True)
            st.image(img_display, use_container_width=True)
            st.markdown('<p style="text-align:center; color:var(--text-muted); font-size:0.8rem; margin-top:5px;">ACTIVE QUERY</p></div>', unsafe_allow_html=True)

            
            if st.button("RUN DEEP LEARNING ANALYSIS", use_container_width=True):


                with st.spinner("Executing KNN Search..."):
                    q_feat = extract_features(img_display, model)
                    distances, indices = knn_model.kneighbors([q_feat])
                
                st.session_state['results'] = (indices[0][1:], distances[0][1:])
        
        if 'results' in st.session_state:
            rec_indices, rec_distances = st.session_state['results']
            st.markdown('<h3 style="margin-top:2rem; color:var(--text-main);">Visual Match Analysis</h3>', unsafe_allow_html=True)

            
            # High-density 6-column grid for smaller results
            grid_cols = st.columns(6)
            for i, idx in enumerate(rec_indices):
                dist = rec_distances[i]
                fpath = filenames[idx]
                with grid_cols[i % 6]:
                    if os.path.exists(fpath):
                        score = dist_to_score(dist)
                        st.markdown(f'''
                        <div class="item-card">
                            <div class="match-badge" style="font-size:0.6rem; padding:0.1rem 0.3rem;">{score}%</div>
                        ''', unsafe_allow_html=True)
                        st.image(Image.open(fpath), use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; border: 2px dashed var(--border); border-radius: 20px;">
            <p style="color:var(--text-muted); font-size:1.2rem;">Ready for input. Upload an image or select one from the browser to begin analysis.</p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    p_col1, p_col2, p_col3 = st.columns([1, 2, 2])
    with p_col1:
        page = st.number_input("Catalog Page", min_value=1, value=1)
    with p_col3:
        st.write("") # Padding
        st.write("") # Padding
        st.caption(f"Showing items {((page-1)*18)+1} to {(page*18)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    page_size = 18
    start = (page-1)*page_size
    end = start + page_size
    files = filenames[start:end]
    
    grid = st.columns(6)
    for i, fpath in enumerate(files):
        with grid[i % 6]:
            if os.path.exists(fpath):
                st.markdown('<div class="item-card">', unsafe_allow_html=True)
                st.image(Image.open(fpath), use_container_width=True)
                if st.button("🎯 USE", key=f"sel_{i}", use_container_width=True):
                    st.session_state.selected_img = fpath
                    st.toast("Item selected for matching!", icon="✅")
                    st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)
