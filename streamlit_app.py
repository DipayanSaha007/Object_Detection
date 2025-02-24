import streamlit as st
from object_from_image import detect_objects_in_image  
from object_from_video import detect_objects_in_video  
import cv2
import os
import time
from ultralytics import YOLO

YOLO('yolov8n.pt')  # Automatically download model if not available

st.title("Object Detection with YOLO")
st.text("This webapp can detect objects from an Image, a Video, or a Webcam stream")
st.text("Use the sidebar to select between Image, Video, and Webcam")

# Sidebar for user options
st.sidebar.title("Options")
option = st.sidebar.selectbox("Choose a mode:", ("Image Detection", "Video Detection", "Webcam Detection"))

if option == "Image Detection":
    st.header("Image Detection")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.read())

        st.text("Processing...")
        progress_bar = st.progress(0)
        for percent in range(0, 101, 20):
            progress_bar.progress(percent)
            time.sleep(0.5)

        st.text("Finalizing image processing...")
        processed_image, detected_objects = detect_objects_in_image("uploaded_image.jpg")

        # Display processed image
        st.image(processed_image, caption="Processed Image", use_container_width=True)

        # Print detected objects with their counts
        if detected_objects:
            st.subheader("Detected Objects")
            for obj, count in detected_objects.items():
                st.write(f"{obj}: {count}")
        else:
            st.write("No objects detected.")

        os.remove("uploaded_image.jpg")

elif option == "Video Detection":
    st.header("Video Detection")
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_video.read())

        st.text("Processing video...")
        cap = cv2.VideoCapture("uploaded_video.mp4")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        progress_bar = st.progress(0)

        detected_objects_summary = {}

        def video_processing_with_progress(video_path):
            global detected_objects_summary
            processed_video_path, detected_objects_summary = detect_objects_in_video(
                video_path,
                progress_callback=lambda processed_frames, frame_results: progress_bar.progress(
                    int((processed_frames / total_frames) * 100)
                ),
            )
            return processed_video_path
        
        processed_video_path = video_processing_with_progress("uploaded_video.mp4")

        if processed_video_path and os.path.exists(processed_video_path):
            st.success("Video processed successfully!")
            with open(processed_video_path, "rb") as file:
                st.download_button(
                    label="Download Processed Video",
                    data=file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )

            # Print detected objects summary
            st.subheader("Detected Objects in Video")
            for obj, count in detected_objects_summary.items():
                st.write(f"{obj}: {count}")

        else:
            st.error("Processed video could not be found!")

        os.remove("uploaded_video.mp4")

elif option == "Webcam Detection":
    st.header("Webcam Detection")
    st.text("Press 'Start' to begin detecting objects from your webcam")

    if st.button("Start"):
        cap = cv2.VideoCapture(0)
        model = YOLO('yolov8n.pt')

        detected_objects_live = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    label = model.names[class_id]
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Count detected objects
                    detected_objects_live[label] = detected_objects_live.get(label, 0) + 1

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame, channels="RGB")

        cap.release()

        # Display detected objects from webcam
        st.subheader("Detected Objects from Webcam")
        for obj, count in detected_objects_live.items():
            st.write(f"{obj}: {count}")
