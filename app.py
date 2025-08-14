import streamlit as st
import json
from torchvision.datasets import STL10
from torchvision.transforms import ToTensor
from PIL import Image, ImageDraw
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

# --- Configuration ---
RETRIEVAL_FOLDER = "retrieval_results"
FAISS_INDEX_FILE = "stl10_vllm_responses.index"
METADATA_FILE = "stl10_vllm_metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"
DATASET_ROOT = "./data"

# --- Helper Functions ---
@st.cache_resource
def load_faiss_index():
    """Loads the FAISS index and metadata."""
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(METADATA_FILE):
        st.error("FAISS index or metadata file not found. Please run the data preparation script.")
        return None, None
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)
    return index, metadata

@st.cache_resource
def load_embedding_model():
    """Loads the SentenceTransformer model."""
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def load_stl10_dataset():
    """Loads the STL-10 test dataset."""
    transform = ToTensor()
    return STL10(root=DATASET_ROOT, split='test', download=True, transform=transform)

# --- Main Streamlit App ---
st.title("🖼️ STL-10 Image Retrieval GUI")

st.markdown("""
Enter a query below to find the most relevant images from the STL-10 dataset based on their generated text descriptions.
""")

# Load resources
index, metadata = load_faiss_index()
model = load_embedding_model()
dataset = load_stl10_dataset()

if index is None or metadata is None:
    st.stop()

# --- Retrieval Section ---
query_text = st.text_input("Enter your query here:", "ocean with clear sky with vehicle")
k_val = st.slider("Number of results to retrieve:", 1, 10, 3)

if st.button("Submit Query"):
    if not query_text:
        st.warning("Please enter a query.")
    else:
        st.info(f"Searching for images related to: **'{query_text}'**")
        with st.spinner("Retrieving images..."):
            try:
                # Encode the query
                query_embedding = model.encode(query_text)
                
                # Search the FAISS index
                D, I = index.search(np.array([query_embedding]).astype("float32"), k=k_val)
                
                st.success(f"Found {k_val} results.")
                
                # Display results
                for i_idx, result_idx in enumerate(I[0]):
                    st.divider()
                    resp_data = metadata[result_idx]
                    image_id_str = resp_data['image_id']
                    image_idx = int(image_id_str.split('_')[1])
                    responses = " ".join(resp_data['responses'].values())
                    
                    # Get the image from the dataset
                    image_tensor, _ = dataset[image_idx]
                    image_pil = Image.fromarray((image_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8'))
                    
                    st.subheader(f"Result {i_idx + 1}")
                    st.image(image_pil, caption=f"Retrieved Image (ID: {image_id_str})")
                    st.markdown(f"**Associated Text Description:**\n> {responses}")

            except Exception as e:
                st.error(f"An error occurred during retrieval: {e}")

st.divider()

# --- STL10 Classes Section ---
st.subheader("STL-10 Dataset Classes")
stl10_classes = [
    "airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"
]
st.write(", ".join(stl10_classes))