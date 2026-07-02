import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Motorrad-Erkennung", layout="wide")
st.title("🏍️ Motorrad-Erkennung mit YOLOv8 (ONNX)")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.onnx")

model = load_model()

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    with st.spinner("Erkenne Objekte..."):
        results = model.predict(img, conf=0.25)

    result = results[0]
    annotated = result.plot()  # zeichnet Boxen ins Bild

    st.image(annotated, caption="Erkennung", channels="BGR", use_container_width=True)

    # Nur Motorräder (COCO-Klasse "motorcycle", id=3) herausfiltern
    names = result.names
    motorrad_treffer = [
        (names[int(box.cls)], float(box.conf))
        for box in result.boxes
        if names[int(box.cls)] == "motorcycle"
    ]

    if motorrad_treffer:
        st.success(f"{len(motorrad_treffer)} Motorrad(räder) gefunden!")
        for name, conf in motorrad_treffer:
            st.write(f"- {name}: {conf:.2%} Konfidenz")
    else:
        st.warning("Kein Motorrad erkannt.")
