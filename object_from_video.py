from ultralytics import YOLO
import cv2
import os

def detect_objects_in_video(video_path):
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Check if video was successfully opened
    if not cap.isOpened():
        print("Error: Couldn't open video file.")
        return None

    # Output video path
    output_path = "processed_video.mp4"

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4

    # Video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Check if VideoWriter is initialized correctly
    if not out.isOpened():
        print("Error: Couldn't initialize video writer.")
        return None

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)

        # Draw results on the frame
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                label = model.names[class_id]

                # Draw bounding box and label
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the processed frame to the output video
        out.write(frame)
        frame_count += 1

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Debug: Check if any frames were written
    if frame_count == 0:
        print("Warning: No frames processed.")
        return None

    print(f"Processed {frame_count} frames.")

    return output_path