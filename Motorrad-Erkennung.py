import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Motorrad-Erkennung", layout="wide")
st.title("🏍️ Motorrad-Erkennung mit YOLOv8 (ONNX – HuggingFace)")

@st.cache_resource
def load_model():
    # ONNX-Modell von HuggingFace – kein Pickle, kein Fehler
    return YOLO("https://huggingface.co/ultralytics/yolov8n-onnx/resolve/main/yolov8n.onnx")

model = load_model()

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    results = model.predict(np.array(image))
    result_image = results[0].plot()

    st.subheader("Ergebnis")
    st.image(result_image, caption="Erkannte Objekte", use_column_width=True)

    st.subheader("Erkannte Klassen")
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        st.write(f"- Klasse: {model.names[cls]} – Sicherheit: {conf:.2f}")
