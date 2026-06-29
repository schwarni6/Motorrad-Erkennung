import streamlit as st
from ultralytics.nn.autobackend import AutoBackend
from PIL import Image
import numpy as np

st.set_page_config(page_title="Motorrad-Erkennung", layout="wide")
st.title("🏍️ Motorrad-Erkennung mit YOLOv8 (lokales ONNX-Modell)")

@st.cache_resource
def load_model():
    return AutoBackend("yolov8n.onnx", device="cpu")  # lokale Datei!

model = load_model()

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    preds = model(img)
    st.write("Modell erfolgreich ausgeführt – Rohdaten:")
    st.write(preds)
