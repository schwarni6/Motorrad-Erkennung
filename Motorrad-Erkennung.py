import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Motorrad Erkennung", layout="wide")

st.title("🏍️ Motorrad-Erkennung mit YOLOv8")

# Modell laden
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # oder dein eigenes Modell

model = load_model()

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # YOLO Inferenz
    results = model.predict(np.array(image))

    # Ausgabe anzeigen
    st.subheader("Ergebnis")
    result_image = results[0].plot()  # erzeugt ein Numpy-Array mit Bounding Boxes
    st.image(result_image, caption="Erkannte Objekte", use_column_width=True)

    # Labels ausgeben
    st.subheader("Erkannte Klassen")
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        st.write(f"- Klasse: {model.names[cls]} – Sicherheit: {conf:.2f}")
