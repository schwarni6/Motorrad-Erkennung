import streamlit as st
import torch
import numpy as np

from PIL import Image, ImageDraw
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

# ---------------------------------------------------

# DEVICE

# ---------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------

# MODELLE LADEN

# ---------------------------------------------------

@st.cache_resource
def load_models():

```
yolo_model = YOLO("yolov8n.pt")

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

return yolo_model, clip_model, clip_processor
```

yolo_model, clip_model, clip_processor = load_models()

# ---------------------------------------------------

# KLASSEN

# ---------------------------------------------------

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

# ---------------------------------------------------

# CLIP ANALYSE

# ---------------------------------------------------

def analyze_crop(crop):

```
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

inputs = {
    k: v.to(device)
    for k, v in inputs.items()
}

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

return (
    best_type,
    best_color,
    type_conf,
    color_conf
)
```

# ---------------------------------------------------

# STREAMLIT UI

# ---------------------------------------------------

st.title("🏍 Motorrad-Erkennung mit YOLO + CLIP")

uploaded_file = st.file_uploader(
"Lade ein Motorradbild hoch",
type=["jpg", "jpeg", "png"]
)

if uploaded_file:

```
# Bild laden
pil_img = Image.open(uploaded_file).convert("RGB")

# Für YOLO als NumPy
img_np = np.array(pil_img)

st.write("Analysiere Bild...")

# YOLO Detection
results = yolo_model(
    img_np,
    classes=[3],   # nur Motorräder
    verbose=False
)

draw = ImageDraw.Draw(pil_img)

motorcycle_found = False

for result in results:

    for box in result.boxes:

        motorcycle_found = True

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # Crop erstellen
        crop = pil_img.crop((x1, y1, x2, y2))

        # CLIP Analyse
        (
            detected_type,
            detected_color,
            type_conf,
            color_conf
        ) = analyze_crop(crop)

        label = (
            f"{detected_color} "
            f"{detected_type}"
        )

        # Rechteck zeichnen
        draw.rectangle(
            [x1, y1, x2, y2],
            outline="green",
            width=4
        )

        # Text zeichnen
        draw.text(
            (x1, y1 - 20),
            label,
            fill="green"
        )

        # Ausgabe
        st.write(f"Erkannt: {label}")
        st.write(
            f"Typ-Sicherheit: "
            f"{type_conf:.2f}"
        )
        st.write(
            f"Farb-Sicherheit: "
            f"{color_conf:.2f}"
        )

if not motorcycle_found:
    st.warning("Kein Motorrad erkannt.")

st.image(
    pil_img,
    caption="Analyse Ergebnis",
    use_column_width=True
)
```
import streamlit as st
import torch
import numpy as np

from PIL import Image, ImageDraw
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

# ---------------------------------------------------

# DEVICE

# ---------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------

# MODELLE LADEN

# ---------------------------------------------------

@st.cache_resource
def load_models():

```
yolo_model = YOLO("yolov8n.pt")

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

return yolo_model, clip_model, clip_processor
```

yolo_model, clip_model, clip_processor = load_models()

# ---------------------------------------------------

# KLASSEN

# ---------------------------------------------------

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

# ---------------------------------------------------

# CLIP ANALYSE

# ---------------------------------------------------

def analyze_crop(crop):

```
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

inputs = {
    k: v.to(device)
    for k, v in inputs.items()
}

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

return (
    best_type,
    best_color,
    type_conf,
    color_conf
)
```

# ---------------------------------------------------

# STREAMLIT UI

# ---------------------------------------------------

st.title("🏍 Motorrad-Erkennung mit YOLO + CLIP")

uploaded_file = st.file_uploader(
"Lade ein Motorradbild hoch",
type=["jpg", "jpeg", "png"]
)

if uploaded_file:

```
# Bild laden
pil_img = Image.open(uploaded_file).convert("RGB")

# Für YOLO als NumPy
img_np = np.array(pil_img)

st.write("Analysiere Bild...")

# YOLO Detection
results = yolo_model(
    img_np,
    classes=[3],   # nur Motorräder
    verbose=False
)

draw = ImageDraw.Draw(pil_img)

motorcycle_found = False

for result in results:

    for box in result.boxes:

        motorcycle_found = True

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # Crop erstellen
        crop = pil_img.crop((x1, y1, x2, y2))

        # CLIP Analyse
        (
            detected_type,
            detected_color,
            type_conf,
            color_conf
        ) = analyze_crop(crop)

        label = (
            f"{detected_color} "
            f"{detected_type}"
        )

        # Rechteck zeichnen
        draw.rectangle(
            [x1, y1, x2, y2],
            outline="green",
            width=4
        )

        # Text zeichnen
        draw.text(
            (x1, y1 - 20),
            label,
            fill="green"
        )

        # Ausgabe
        st.write(f"Erkannt: {label}")
        st.write(
            f"Typ-Sicherheit: "
            f"{type_conf:.2f}"
        )
        st.write(
            f"Farb-Sicherheit: "
            f"{color_conf:.2f}"
        )

if not motorcycle_found:
    st.warning("Kein Motorrad erkannt.")

st.image(
    pil_img,
    caption="Analyse Ergebnis",
    use_column_width=True
)
```
