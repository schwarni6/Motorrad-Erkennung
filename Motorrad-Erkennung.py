import streamlit as st
from ultralytics.nn.autobackend import AutoBackend
from PIL import Image
import numpy as np

st.set_page_config(page_title="Motorrad-Erkennung", layout="wide")
st.title("🏍️ Motorrad-Erkennung mit YOLOv8 (ONNX – Direkt geladen)")

@st.cache_resource
def load_model():
    model_path = "https://huggingface.co/ultralytics/yolov8n-onnx/resolve/main/yolov8n.onnx"
    return AutoBackend(model_path, device="cpu")

model = load_model()

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    preds = model(img)
    # preds enthält die rohen ONNX-Ausgaben
    # Ultralytics hat keine Plot-Funktion für AutoBackend → wir müssen selbst zeichnen

    st.write("Modell erfolgreich ausgeführt – Rohdaten:")
    st.write(preds)
