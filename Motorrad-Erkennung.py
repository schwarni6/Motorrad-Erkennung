import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Motorrad-Erkennung", layout="wide")
st.title("🏍️ Motorrad-Erkennung mit YOLOv8")

# WICHTIG: kaputte Datei löschen, falls vorhanden
if os.path.exists("yolov8n.pt"):
    os.remove("yolov8n.pt")

def load_model():
    # lädt automatisch das echte Modell herunter
    return YOLO("yolov8n.pt")

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
