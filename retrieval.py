import json
import torch
from torchvision.datasets import STL10
from torchvision.transforms import ToTensor
from PIL import Image, ImageDraw
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from dataclasses import asdict
from typing import Optional, NamedTuple
from vllm.engine.arg_utils import EngineArgs
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

def save_image_with_text(image_idx,image_obj, query_text, response_text, retrieval_folder="retrieval_results"):
    """
    Saves a PIL Image object with the query and response text.
    """
    os.makedirs(retrieval_folder, exist_ok=True)
    draw = ImageDraw.Draw(image_obj)
    
    # Simple text layout to avoid overlap
    draw.text((2, 200), f"{query_text}", fill="white")
    # draw.text((10, 40), f"Response: {response_text}", fill="white")
    
    # Use the response text as part of the filename for uniqueness
    sanitized_response = response_text[:30].replace("/", "_").replace("\\", "_")
    image_obj.save(f"{retrieval_folder}/retrieved_{image_idx}.png")
    print(f"Saved image to {retrieval_folder}/retrieved_{image_idx}.png")


def load_stl10_dataset(root_dir="./data", download=True):
    """
    Loads the STL-10 test dataset.
    """
    transform = ToTensor()
    return STL10(root=root_dir, split='test', download=download, transform=transform)


# load the databases
index = faiss.read_index("stl10_vllm_responses.index")
with open("stl10_vllm_metadata.json", "r") as f:
    metadata = json.load(f)

# load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# load the STL10 DATASET
dataset = load_stl10_dataset()

# provide a query, and retrieve the most similar responses, and the image based on that. Save the image with the provided query
query = "ocean with clear sky with vehicle"
query_embedding = model.encode(query)

# Search for the most similar responses (top k)
D, I = index.search(np.array([query_embedding]).astype("float32"), k=3)

print("Most similar responses:")

for i in I[0]:
    resp_data = metadata[i]
    image_id_str = resp_data['image_id']
    image_idx = int(image_id_str.split('_')[1])
    
    # Get the responses generated for this image
    responses = resp_data['responses']
    
    # Get the image from the dataset using the extracted index
    image_tensor, _ = dataset[image_idx]
    image_pil = Image.fromarray((image_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8'))
    # pad the image with black background to make it bigger, so that text annotation can fit
    padded_image = Image.new("RGB", (image_pil.width + 100, image_pil.height + 200), (0, 0, 0))
    padded_image.paste(image_pil, (50, 50))

    # Print the responses and save the image
    print(f"\nImage ID: {image_id_str}")
    print(f"Responses: {responses}")
    
    # Concatenate all responses for a more descriptive filename
    all_responses = " ".join(responses.values())
    save_image_with_text(image_idx,padded_image, query, all_responses)