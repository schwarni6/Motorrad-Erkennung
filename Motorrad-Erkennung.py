import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Motorrad-Erkennung", layout="wide")
st.title("🏍️ Motorrad-Erkennung mit YOLOv8")

@st.cache_resource
def load_model():
    # Lädt automatisch die offiziellen, geprüften Gewichte herunter (COCO-trainiert)
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    with st.spinner("Erkenne Objekte..."):
        results = model.predict(img, conf=0.25)

    result = results[0]
    annotated = result.plot()  # Bild mit eingezeichneten Boxen

    st.image(annotated, caption="Erkennung", channels="BGR", use_container_width=True)

    names = result.names
    treffer = [
        (names[int(box.cls)], float(box.conf))
        for box in result.boxes
        if names[int(box.cls)] == "motorcycle"
    ]

    if treffer:
        st.success(f"{len(treffer)} Motorrad(räder) gefunden!")
        for name, conf in treffer:
            st.write(f"- {name}: {conf:.2%} Konfidenz")
    else:
        st.warning("Kein Motorrad erkannt.")
