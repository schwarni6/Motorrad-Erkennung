import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# YOLO
yolo_model = YOLO("yolov8n.pt")

# CLIP
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

motorcycle_types = [
    "sport motorcycle",
    "chopper motorcycle",
    "adventure touring motorcycle",
    "vespa scooter",
    "motocross dirt bike"
]

colors = [
    "black",
    "red",
    "blue",
    "white",
    "green",
    "yellow",
    "grey"
]


def analyze_crop(crop):

    prompts = (
        [f"a photo of a {m}" for m in motorcycle_types] +
        [f"a {c} motorcycle" for c in colors]
    )

    inputs = clip_processor(
        text=prompts,
        images=crop,
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = clip_model(**inputs)

    logits = outputs.logits_per_image[0]

    type_logits = logits[:len(motorcycle_types)]
    color_logits = logits[len(motorcycle_types):]

    type_probs = type_logits.softmax(dim=-1)
    color_probs = color_logits.softmax(dim=-1)

    type_idx = type_probs.argmax().item()
    color_idx = color_probs.argmax().item()

    best_type = motorcycle_types[type_idx]
    best_color = colors[color_idx]

    type_conf = type_probs[type_idx].item()
    color_conf = color_probs[color_idx].item()

    return best_type, best_color, type_conf, color_conf
