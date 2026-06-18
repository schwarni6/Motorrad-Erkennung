import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Motorrad-Erkennung", layout="wide")
st.title("🏍️ Motorrad-Erkennung mit YOLOv8")

@st.cache_resource
def load_model():
    # Wichtig:
    # 1. KEINE lokale yolov8n.pt Datei im Repo haben
    # 2. Dann lädt Ultralytics das offizielle Modell automatisch herunter
    model = YOLO("yolov8n.pt")
    return model

model = load_model()

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Inferenz
    img_array = np.array(image)
    results = model.predict(img_array)

    st.subheader("Ergebnis")
    result_image = results[0].plot()
    st.image(result_image, caption="Erkannte Objekte", use_column_width=True)

    st.subheader("Erkannte Klassen")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        st.write(f"- Klasse: {model.names[cls_id]} – Sicherheit: {conf:.2f}")
