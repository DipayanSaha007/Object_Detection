import streamlit as st
from object_from_image import detect_objects_in_image  # Function for image detection
from object_from_video import detect_objects_in_video  # Function for video detection
import cv2
import os
import time
from ultralytics import YOLO
from collections import Counter

# Load YOLO model
model = YOLO('yolov8n.pt')

st.title("Object Detection with YOLO")
st.text("This webapp can detect objects from an Image, a Video, or a Webcam stream.")
st.text("Use the sidebar to select between Image, Video, and Webcam.")

# Sidebar for user options
st.sidebar.title("Options")
option = st.sidebar.selectbox("Choose a mode:", ("Image Detection", "Video Detection", "Webcam Detection"))

def get_object_counts(results):
    """Extract object counts from YOLO results."""
    object_counts = Counter()
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Object class index
            label = model.names[class_id]  # Get object name
            object_counts[label] += 1  # Count occurrences
    
    return object_counts

if option == "Image Detection":
    st.header("Image Detection")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.read())
        
        st.text("Processing")
        progress_bar = st.progress(0)
        for percent in range(0, 101, 20):
            progress_bar.progress(percent)
            time.sleep(0.5)

        # Detect objects
        results = model("uploaded_image.jpg")
        processed_image = results[0].plot()

        # Get object counts
        object_counts = get_object_counts(results)

        # Display processed image
        st.image(processed_image, caption="Processed Image", use_column_width=True)
        
        # Display object counts
        st.subheader("Detected Objects:")
        for obj, count in object_counts.items():
            st.write(f"**{obj}: {count}**")

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

        object_counts = Counter()  # Store total detected object counts

        def video_processing_with_progress(video_path):
            def progress_callback(processed_frames, frame_results):
                # Update progress bar
                progress_bar.progress(int((processed_frames / total_frames) * 100))
                
                # Count detected objects in the frame
                frame_object_counts = get_object_counts(frame_results)
                object_counts.update(frame_object_counts)

            processed_video_path = detect_objects_in_video(video_path, progress_callback)
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

            # Display object counts
            st.subheader("Detected Objects in Video:")
            for obj, count in object_counts.items():
                st.write(f"**{obj}: {count}**")

        else:
            st.error("Processed video could not be found!")

        os.remove("uploaded_video.mp4")

elif option == "Webcam Detection":
    st.header("Webcam Detection")
    st.text("Press 'Start' to begin detecting objects from your webcam")

    if st.button("Start"):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            processed_frame = results[0].plot()

            # Get object counts
            object_counts = get_object_counts(results)

            # Display detected objects in real-time
            st.image(processed_frame, channels="RGB")
            st.subheader("Objects Detected in Current Frame:")
            for obj, count in object_counts.items():
                st.write(f"**{obj}: {count}**")

        cap.release()
