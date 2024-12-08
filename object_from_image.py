from ultralytics import YOLO
import cv2

def detect_objects_in_image(image_path):
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Load the image
    image = cv2.imread(image_path)

    # Perform inference
    results = model(image)

    # Draw results on the image
    for result in results:
        boxes = result.boxes  # Detected bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            label = model.names[class_id]

            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convert the processed image to RGB (for Streamlit display)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image