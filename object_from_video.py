from ultralytics import YOLO
import cv2
import tempfile

def detect_objects_in_video(video_path, progress_callback=None):
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Couldn't open video file.")
        return None

    # Create temporary output file
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = temp_output.name
    temp_output.close()

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize VideoWriter
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)

        # Draw bounding boxes and labels
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                label = model.names[class_id]
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write processed frame to output
        out.write(frame)

        # Update progress
        processed_frames += 1
        if progress_callback:
            progress_callback(processed_frames)

    cap.release()
    out.release()

    return output_path
