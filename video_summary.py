import os
import json
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from dataclasses import asdict
from typing import Optional, NamedTuple
from vllm.lora.request import LoRARequest
import numpy as np
from transformers import AutoProcessor, AutoTokenizer

# set cuda devices to specific gpus
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # Set to the GPU IDs you want to use

def get_front3_stitched_image(video_dir,frame_id):
    front_views = [
        np.array(Image.open(os.path.join(video_dir, f"FRONT_LEFT", f"{frame_id}.png"))),
        np.array(Image.open(os.path.join(video_dir, f"FRONT", f"{frame_id}.png"))),
        np.array(Image.open(os.path.join(video_dir, f"FRONT_RIGHT", f"{frame_id}.png"))),
    ]
    stitched_front = np.concatenate(front_views, axis=1)
    # resize the stitched image to 224x224
    stitched_front = np.array(Image.fromarray(stitched_front).resize((224, 224)))

    return stitched_front

def get_back3_stitched_image(video_dir,frame_id):
    back_views = [
        np.array(Image.open(os.path.join(video_dir, f"REAR_RIGHT", f"{frame_id}.png"))),
        np.array(Image.open(os.path.join(video_dir, f"REAR", f"{frame_id}.png"))),
        np.array(Image.open(os.path.join(video_dir, f"REAR_LEFT", f"{frame_id}.png"))),
    ]
    stitched_back = np.concatenate(back_views, axis=1)
    # resize the stitched image to 224x224
    stitched_back = np.array(Image.fromarray(stitched_back).resize((224, 224)))
    return stitched_back


video_dir = "/datassd4/users/hardik/dev/waymo_e2e/data/0b8097d37ac2cc7a832e2978f431f843"
frame_ids = ["010", "050"]
stitched_images = []
for frame_id in frame_ids:
    front_img = get_front3_stitched_image(video_dir,frame_id)
    # back_img = get_back3_stitched_image(video_dir,frame_id)
    stitched_images.append(Image.fromarray(front_img))
    # stitched_images.append(Image.fromarray(back_img))

print(f"Total stitched images: {len(stitched_images)}" )

# Add 10 <image> tokens, one for each stitched frame
image_placeholders = "\n".join([f"<image>" for i in range(len(stitched_images))])

SEGMENT_PROMPT = f"""[INST]{image_placeholders} You are a driving examiner evaluating a 5-second driving video segment. The video consists of 2 consecutive stitched front views from the ego vehicle, sampled at 2 frames per 5 second.
The images are provided above. Based on the full sequence of images, write an observer’s note that addresses:
1. Ego vehicle control: lane position, spacing, following distance.
2. Compliance: traffic signs, signals, right-of-way.
3. Maneuvers: turns, lane changes, or stops.
4. Hazards: any potential risks or conflicts.
5. Overall driving performance for this short segment.
Write the response as if it were part of a DMV behind the wheel driving performance evaluation sheet.
Be concise but specific, describing what happens across the 5 seconds.
[/INST]"""


# Setup VLLM engine
engine_args = EngineArgs(
    model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    max_model_len=32768,
    limit_mm_per_prompt={"image": len(stitched_images)},  # we are passing 5 images
    # tensor_parallel_size=7,  # Use 4 GPUs for tensor parallelism
)
llm = LLM(**asdict(engine_args))

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.2,
    max_tokens=512
)

# Prepare input (all 5 images in one prompt)
inputs = [{
    "prompt": SEGMENT_PROMPT,
    "multi_modal_data": {
        "image": stitched_images
    }
}]

outputs = llm.generate(inputs, sampling_params=sampling_params)

# Extract text
note = outputs[0].outputs[0].text.strip()
print("=== DMV Observer Note ===")
print(note)

# Save to JSON
results = {"video_id": os.path.basename(video_dir), "note": note}
with open("waymo_dmv_notes.json", "w") as f:
    json.dump(results, f, indent=4)
