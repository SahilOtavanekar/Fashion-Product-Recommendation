# Fashion Product Recommendation System

A deep learning–based fashion product recommendation system using ResNet50 feature extraction and k-Nearest Neighbors similarity search.

## Features

- 👗 **Deep Learning–Powered Recommendations**: Uses ResNet50 CNN to extract image features
- 🔍 **Semantic Search**: Find similar fashion products based on image similarity
- 🎨 **Modern UI**: Built with Streamlit for an elegant, user-friendly interface
- ⚡ **Fast Processing**: Efficient feature extraction and similarity matching

## Tech Stack

- **Backend**: Python, TensorFlow/Keras, scikit-learn
- **Frontend**: Streamlit
- **Deep Learning**: ResNet50 (pre-trained CNN)
- **ML**: k-Nearest Neighbors for similarity matching

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Fashion-Product-Recommendation.git
cd Fashion-Product-Recommendation
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/Scripts/activate  # On Windows
# or
source .venv/bin/activate      # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
├── app.py                                 # Main Streamlit application
├── Fashion_Product_Recommendation.ipynb   # Jupyter notebook with analysis
├── requirements.txt                       # Python dependencies
├── fix_pickle.py                          # Utility for pickle file handling
├── filenames.pkl                          # Pickle file with image filenames
├── Images_features.pkl                    # Pickle file with extracted features
└── images/                                # Directory containing product images
```

## How It Works

1. **Feature Extraction**: ResNet50 extracts 2048-dimensional feature vectors from images
2. **Similarity Matching**: k-NN algorithm finds the most similar products in the database
3. **Recommendations**: Returns top-N similar products based on image features

## Dependencies

- streamlit >= 1.32.0
- tensorflow >= 2.13.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- Pillow >= 10.0.0

## Authors

Sahil Otavanekar

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
