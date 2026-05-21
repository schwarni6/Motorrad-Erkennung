import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

# 1. Modelle laden
# YOLOv8 für die Erkennung des Motorrads im Bild
yolo_model = YOLO("yolov8n.pt")  # 'n' ist die leichteste/schnellste Version

# CLIP für die Klassifizierung von Modell und Farbe (Zero-Shot)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Definition der Klassen, nach denen gesucht werden soll
# (Erweitere diese Listen je nach Bedarf)
motorcycle_models = ["Sportbike", "Chopper", "Adventure Tourer", "Vespa Scooter", "Motocross"]
colors = ["black", "red", "blue", "white", "green", "yellow", "grey"]

def analyze_crop(image_crop, candidate_labels):
    """Nutzt CLIP, um das passendste Label aus einer Liste für den Bildausschnitt zu finden."""
    inputs = clip_processor(text=candidate_labels, images=image_crop, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    
    # Wahrscheinlichkeiten berechnen
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=-1)
    
    # Index des höchsten Wertes finden
    max_idx = probs.argmax().item()
    return candidate_labels[max_idx]

def detect_motorcycle_details(image_path):
    # Bild laden
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # YOLO Erkennung ausführen
    results = yolo_model(img)

    for result in results:
        for box in result.boxes:
            # Klasse 3 in COCO ist "motorcycle"
            class_id = int(box.cls[0])
            if class_id == 3:
                # Koordinaten der Bounding Box extrahieren
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Motorrad ausschneiden
                crop = pil_img.crop((x1, y1, x2, y2))
                
                # Modell und Farbe via CLIP bestimmen
                detected_model = analyze_crop(crop, motorcycle_models)
                detected_color = analyze_crop(crop, colors)
                
                print(f"Erkannt: {detected_color} {detected_model}")

                # Ergebnisse in das Originalbild einzeichnen (für die Anzeige)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f"{detected_color} {detected_model}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Ergebnis anzeigen
    cv2.imshow("Motorrad Erkennung", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Beispielaufruf (Ersetze 'motorrad.jpg' mit deinem Bildpfad)
if __name__ == "__main__":
    # detect_motorcycle_details("dein_test_bild.jpg")
    pass
