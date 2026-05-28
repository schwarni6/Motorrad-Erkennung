import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

# 1. Modelle laden
# YOLOv8 für die Erkennung des Motorrads im Bild
yolo_model = YOLO("yolov8n.pt")

# CLIP für die Klassifizierung von Modell und Farbe (Zero-Shot)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Definition der Klassen
motorcycle_models = ["Sportbike", "Chopper", "Adventure Tourer", "Vespa Scooter", "Motocross"]
colors = ["black", "red", "blue", "white", "green", "yellow", "grey"]

def analyze_crop_efficient(image_crop, model_labels, color_labels):
    """
    Verarbeitet das Bild nur EINMAL in CLIP und bestimmt 
    sowohl Modell als auch Farbe mithilfe von Prompt Engineering.
    """
    # Prompt Engineering für bessere CLIP-Ergebnisse
    model_prompts = [f"a photo of a {label}" for label in model_labels]
    color_prompts = [f"a motorcycle of {label} color" for label in color_labels]
    
    # Alle Prompts kombinieren
    all_prompts = model_prompts + color_prompts
    
    # Bild und alle Texte gleichzeitig an den Prozessor übergeben
    inputs = clip_processor(text=all_prompts, images=image_crop, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = clip_model(**inputs)
    
    # Logits extrahieren (Ähnlichkeit zwischen Bild und Texten)
    logits_per_image = outputs.logits_per_image[0] # Tensor für das erste (und einzige) Bild
    
    # Logits wieder in Modell- und Farbbereich aufteilen
    model_logits = logits_per_image[:len(model_prompts)]
    color_logits = logits_per_image[len(model_prompts):]
    
    # Softmax separat anwenden, um Wahrscheinlichkeiten zu erhalten
    model_probs = model_logits.softmax(dim=-1)
    color_probs = color_logits.softmax(dim=-1)
    
    # Beste Treffer ermitteln
    best_model = model_labels[model_probs.argmax().item()]
    best_color = color_labels[color_probs.argmax().item()]
    
    return best_model, best_color

def detect_motorcycle_details(image_path):
    # Bild laden
    img = cv2.imread(image_path)
    if img is None:
        print(f"Fehler: Bild unter {image_path} konnte nicht geladen werden.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # YOLO Erkennung ausführen
    results = yolo_model(img, verbose=False) # verbose=False unterdrückt den Standard-YOLO-Output

    motorcycle_detected = False

    for result in results:
        for box in result.boxes:
            # Klasse 3 in COCO ist "motorcycle"
            class_id = int(box.cls[0])
            if class_id == 3:
                motorcycle_detected = True
                
                # Koordinaten der Bounding Box extrahieren
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Motorrad ausschneiden
                crop = pil_img.crop((x1, y1, x2, y2))
                
                # Modell und Farbe via optimiertem CLIP-Aufruf bestimmen
                detected_model, detected_color = analyze_crop_efficient(crop, motorcycle_models, colors)
                
                print(f"Erkannt: {detected_color} {detected_model}")

                # Ergebnisse in das Originalbild einzeichnen
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f"{detected_color} {detected_model}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if not motorcycle_detected:
        print("Kein Motorrad im Bild gefunden.")

    # Ergebnis anzeigen
    cv2.imshow("Motorrad Erkennung", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    for i in range(5): cv2.waitKey(1) # Behebt eventuelle GUI-Hänger unter macOS/Linux

if __name__ == "__main__":
    # Aktiviere den Aufruf und setze deinen Testbild-Pfad ein
    detect_motorcycle_details("dein_test_bild.jpg")
