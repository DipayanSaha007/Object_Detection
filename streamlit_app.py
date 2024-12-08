import streamlit as st
from object_from_image import detect_objects_in_image  # Ensure this function handles image detection
from object_from_video import detect_objects_in_video  # Ensure this function handles video detection
import cv2
import os
import time

st.title("Object Detection with YOLO")
st.text("This webapp can detect object from a Image or a Video")
st.text("Use the sidebar to select between Image and Video")
# Sidebar for user options
st.sidebar.title("Options")
option = st.sidebar.selectbox("Choose a mode:", ("Image Detection", "Video Detection"))

if option == "Image Detection":
    st.header("Image Detection")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        # Save the uploaded file
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.read())
        
        # Initialize progress bar
        st.text("Processing Image...")
        progress_bar = st.progress(0)
        
        # Simulate image processing time
        for percent in range(0, 101, 20):
            progress_bar.progress(percent)
            time.sleep(0.5)  # Simulate time delay
        
        # Perform object detection
        st.text("Finalizing image processing...")
        processed_image = detect_objects_in_image("uploaded_image.jpg")
        
        # Display results
        st.image(processed_image, caption="Processed Image", use_container_width=True)
        
        # Clean up uploaded image
        os.remove("uploaded_image.jpg")

elif option == "Video Detection":
    st.header("Video Detection")
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        # Save the uploaded video
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
        
        st.text("Processing Video...")
        
        # Open the video to calculate total frames
        cap = cv2.VideoCapture("uploaded_video.mp4")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Initialize progress bar
        progress_bar = st.progress(0)

        # Process the video and get the output path
        def video_processing_with_progress(video_path):
            processed_video_path = detect_objects_in_video(
                video_path,
                progress_callback=lambda processed_frames: progress_bar.progress(
                    int((processed_frames / total_frames) * 100)
                ),
            )
            return processed_video_path

        # Call the video processing function with progress updates
        processed_video_path = video_processing_with_progress("uploaded_video.mp4")

        # Check if the file exists and display it
        if processed_video_path and os.path.exists(processed_video_path):
            st.success("Video processed successfully!")
            
            # Provide a download link
            with open(processed_video_path, "rb") as file:
                st.download_button(
                    label="Download Processed Video",
                    data=file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
        else:
            st.error("Processed video could not be found!")
        
        # Clean up uploaded video
        os.remove("uploaded_video.mp4")
