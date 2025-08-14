import json
import torch
from torchvision.datasets import STL10
from torchvision.transforms import ToTensor
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from dataclasses import asdict
from typing import Optional, NamedTuple
from vllm.engine.arg_utils import EngineArgs
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Define the prompts for the VLLM
PROMPTS = [
    "what are the main candidates in the image?",
    "what is the background mainly about?",
    "what activities are present in the image?"
]

class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    stop_token_ids: Optional[list[int]] = None
    lora_requests: Optional[list[LoRARequest]] = None

def run_llava_next(questions: list[str]) -> ModelRequestData:
    """
    Sets up the engine arguments and formats prompts for LLaVA-1.6.
    """
    modality = "image"
    prompts = [f"[INST] <image>\n{question} [/INST]" for question in questions]
    engine_args = EngineArgs(
        model="llava-hf/llava-v1.6-mistral-7b-hf",
        max_model_len=8192,
        limit_mm_per_prompt={modality: 1},
    )
    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )

def load_stl10_dataset(root_dir="./data", download=True):
    """
    Loads the STL-10 test dataset.
    """
    transform = ToTensor()
    return STL10(root=root_dir, split='test', download=download, transform=transform)

def save_to_json(data, filename="stl10_vllm_responses.json"):
    """
    Saves the dictionary to a JSON file.
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to {filename}")

def build_vector_db(results, prompts, embedding_model_name="all-MiniLM-L6-v2"):
    """
    Converts prompts and responses into embeddings and stores them in a FAISS vector DB.
    Returns the FAISS index and a metadata list.
    """
    model = SentenceTransformer(embedding_model_name)
    embeddings = []
    metadata = []

    for image_id, resp_dict in results.items():
        # Concatenate prompts and responses for each image
        concat_text = " ".join([f"{p} {resp_dict[p]}" for p in prompts])
        emb = model.encode(concat_text)
        embeddings.append(emb)
        metadata.append({"image_id": image_id, "responses": resp_dict})

    embeddings = np.vstack(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, metadata

def main():
    """
    Main function to orchestrate the process.
    """
    # 1. Load the STL-10 dataset (test split)
    print("Loading STL-10 dataset...")
    dataset = load_stl10_dataset()
    print("Dataset loaded successfully.")

    # 2. Get the model configuration and prompts for LLaVA-1.6
    req_data = run_llava_next(PROMPTS)
    
    engine_args = asdict(req_data.engine_args)
    # The image input type is handled internally by the specific VLLM model class.
    llm = LLM(**engine_args)

    # 3. Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100
    )

    # Dictionary to store the results
    results = {}

    # 4. Process each image in the dataset
    print("Starting forward pass on images...")
    for i in range(len(dataset)):
        # Get image tensor and convert to PIL Image
        image_tensor, _ = dataset[i]
        image_pil = Image.fromarray((image_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8'))

        # Prepare the requests for the VLLM
        inputs = []
        for prompt in req_data.prompts:
            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image_pil
                }
            })

        # Perform the forward pass
        try:
            outputs = llm.generate(
                inputs,
                sampling_params=sampling_params,
                lora_request=req_data.lora_requests
            )
        except Exception as e:
            print(f"Error during forward pass for image {i}: {e}")
            continue

        # Extract responses and store them
        image_id = f"image_{i}"
        responses_for_image = {}
        for j, output in enumerate(outputs):
            # LLaVA-1.6's prompt format includes the instruction which we need to strip
            generated_text = output.outputs[0].text.strip()
            responses_for_image[PROMPTS[j]] = generated_text
        
        results[image_id] = responses_for_image
        print(f"Processed image {i}/{len(dataset)-1}")
        # if i == 100:
        #     break

    # 5. Save the results to a JSON file
    save_to_json(results)

    # 6. Store embeddings and metadata for retrieval
    # Build the vector database and metadata
    index, metadata = build_vector_db(results, PROMPTS)

    # Save the FAISS index and metadata for later retrieval
    faiss.write_index(index, "stl10_vllm_responses.index")
    with open("stl10_vllm_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print("Vector DB and metadata saved for retrieval.")

    
if __name__ == "__main__":
    main()