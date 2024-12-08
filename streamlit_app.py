import streamlit as st
from object_from_image import detect_objects_in_image  # Ensure this function handles image detection
from object_from_video import detect_objects_in_video  # Ensure this function handles video detection
import os
import time  # For simulating progress updates

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
        
        # Perform object detection
        st.text("Processing image...")
        progress_bar = st.progress(0)  # Initialize progress bar
        for percent in range(0, 101, 20):  # Simulating progress
            time.sleep(0.1)  # Simulate processing delay
            progress_bar.progress(percent)
        
        processed_image = detect_objects_in_image("uploaded_image.jpg")
        
        # Display results
        st.image(processed_image, caption="Processed Image", use_container_width=True)
        
        # Clean up uploaded image
        os.remove("uploaded_image.jpg")
        st.success("Image processing complete!")

elif option == "Video Detection":
    st.header("Video Detection")
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        # Save the uploaded video
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
        
        st.text("Processing video...")
        
        # Initialize progress bar
        progress_bar = st.progress(0)
        for percent in range(0, 101, 10):  # Simulating progress for video processing
            time.sleep(0.3)  # Simulate processing delay
            progress_bar.progress(percent)
        
        # Process the video and get the output path
        processed_video_path = detect_objects_in_video("uploaded_video.mp4")

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
        st.success("Video processing complete!")