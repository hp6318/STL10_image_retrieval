import json
import torch
from torchvision.datasets import STL10
from torchvision.transforms import ToTensor
from PIL import Image, ImageDraw, ImageFont
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from dataclasses import asdict
from typing import Optional, NamedTuple
from vllm.engine.arg_utils import EngineArgs
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

from sklearn.metrics import accuracy_score, confusion_matrix
import re
import os

# Define the prompts for the VLLM
PROMPTS = [
    "You are a classification model. Give a class label to this image from 'airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'"
]
# Define the class labels and a mapping from label to index
CLASS_LABELS = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
LABEL_TO_INDEX = {label: i for i, label in enumerate(CLASS_LABELS)}

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


def main():
    """
    Main function to orchestrate the process.
    """
    # 1. Load the STL-10 dataset (test split)
    print("Loading STL-10 dataset...")
    dataset = load_stl10_dataset()
    print("Dataset loaded successfully. Total samples:", len(dataset))

    # # 2. Get the model configuration and prompts for LLaVA-1.6
    # req_data = run_llava_next(PROMPTS)
    
    # engine_args = asdict(req_data.engine_args)
    # # The image input type is handled internally by the specific VLLM model class.
    # llm = LLM(**engine_args)

    # # 3. Set up sampling parameters
    # sampling_params = SamplingParams(
    #     temperature=0.7,
    #     top_p=0.9,
    #     max_tokens=100
    # )
    
    # # Lists to store ground truth and predicted labels
    # ground_truth_labels = []
    # predicted_labels = []
    # results = {}

    # # 4. Process each image in the dataset
    # print("Starting forward pass on images...")
    # for i in range(len(dataset)):
    #     # Get image tensor and convert to PIL Image
    #     image_tensor, ground_truth_idx = dataset[i]
    #     image_pil = Image.fromarray((image_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8'))

    #     # Store the ground truth label
    #     ground_truth_labels.append(CLASS_LABELS[ground_truth_idx])

    #     # Prepare the requests for the VLLM
    #     inputs = []
    #     for prompt in req_data.prompts:
    #         inputs.append({
    #             "prompt": prompt,
    #             "multi_modal_data": {
    #                 "image": image_pil
    #             }
    #         })

    #     # Perform the forward pass
    #     try:
    #         outputs = llm.generate(
    #             inputs,
    #             sampling_params=sampling_params,
    #             lora_request=req_data.lora_requests
    #         )
    #     except Exception as e:
    #         print(f"Error during forward pass for image {i}: {e}")
    #         predicted_labels.append(None) 
    #         continue
        
    #     # Extract the generated text
    #     generated_text = outputs[0].outputs[0].text.strip().lower()

    #     # Regular expression to find the class label in the text
    #     match = re.search(r'\b(?:' + '|'.join(CLASS_LABELS) + r')\b', generated_text)
        
    #     # Determine the predicted label
    #     if match:
    #         predicted_label = match.group(0)
    #     else:
    #         predicted_label = "unknown"  # Or another placeholder for unclassified images

    #     # Store the predicted label
    #     predicted_labels.append(predicted_label)
    #     # Store the results
    #     image_id = f"image_{i}"
    #     responses_for_image = {
    #         PROMPTS[0]: generated_text,
    #         "ground_truth": CLASS_LABELS[ground_truth_idx],
    #         "predicted_label": predicted_label
    #     }
    #     results[image_id] = responses_for_image

    #     print(f"Processed image {i}/{len(dataset)-1}")
        

    # save_to_json(results,"stl10_vllm_clf.json")
    # # Compute and print evaluation metrics
    # print("\n" + "="*50)
    # print("Computing Evaluation Metrics...")
    
    # # Filter out samples where prediction failed
    # valid_predictions = [pred for pred in predicted_labels if pred in CLASS_LABELS]
    # valid_ground_truths = [gt for gt, pred in zip(ground_truth_labels, predicted_labels) if pred in CLASS_LABELS]

    # if not valid_predictions:
    #     print("No valid predictions were made. Cannot compute metrics.")
    #     return

    # # Compute Classification Accuracy
    # accuracy = accuracy_score(valid_ground_truths, valid_predictions)
    # print(f"Classification Accuracy: {accuracy:.4f}")

    # # Compute Confusion Matrix
    # cm = confusion_matrix(valid_ground_truths, valid_predictions, labels=CLASS_LABELS)
    # print("\nConfusion Matrix:")
    # print(cm)
    
    # print("="*50)

    # # save those images in misclassified/ which were misclassified or have unknown pred label. Write  on the image pred and gt in small fonts
    # os.makedirs("misclassified", exist_ok=True)
    # for i in range(len(dataset)):
    #     if predicted_labels[i] != ground_truth_labels[i] or predicted_labels[i] == "unknown":
    #         image_tensor, _ = dataset[i]
    #         image_pil = Image.fromarray((image_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8'))
    #         draw = ImageDraw.Draw(image_pil)
    #         draw.text((5, 5), f"GT: {ground_truth_labels[i]}", fill="red")
    #         draw.text((5, 20), f"Pred: {predicted_labels[i]}", fill="green")
    #         image_pil.save(f"misclassified/image_{i}_{predicted_labels[i]}_{ground_truth_labels[i]}.png")

if __name__ == "__main__":
    main()