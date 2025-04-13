import streamlit as st
import cv2
import numpy as np
import requests
import os

def skin_tone_bias_face_detection():
    st.title("Skin Tone Bias in Face Detection")
    model_choice = st.selectbox("Select Model", ["OpenCV", "MTCNN", "dlib"], index=0)
    input_method = st.selectbox("Select Input Method", ["Upload Image", "Use Image URL"], index=0)

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
        else:
            st.write("No image uploaded. Using default image.")
            default_path = os.path.join("default_images", "group_photo.jpg")
            image = cv2.imread(default_path)
    else:
        img_url = st.text_input("Enter image URL:")
        if img_url:
            image = cv2.imdecode(np.asarray(bytearray(requests.get(img_url).content), dtype=np.uint8), 1)
        else:
            st.write("No URL provided. Using default image.")
            default_path = os.path.join("default_images", "group_photo.jpg")
            image = cv2.imread(default_path)

    if model_choice == "OpenCV":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Detected Faces", use_column_width=True)
    else:
        st.write(f"Model {model_choice} support is under development.")

if __name__ == "__main__":
    skin_tone_bias_face_detection()
