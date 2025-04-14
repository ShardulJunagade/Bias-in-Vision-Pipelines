import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import os
import requests
from io import BytesIO
from PIL import ImageDraw, ImageOps

# For MTCNN
from facenet_pytorch import MTCNN

# For dlib
import dlib

#############################################
# Cache the MTCNN model
@st.cache_resource
def load_mtcnn_model():
    return MTCNN(keep_all=True)

# Cache the dlib face detector
@st.cache_resource
def load_dlib_detector():
    return dlib.get_frontal_face_detector()



#############################################
# OpenCV Detection
def detect_faces_opencv(image):
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    result = []
    for idx, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2) 
        cv2.putText(cv_image, f"Face {idx+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) 
        result.append({
            "Face ID": f"Face {idx+1}",
            "X": x,
            "Y": y,
            "W": w,
            "H": h,
            "Confidence": "N/A"  # OpenCV Haar doesn't return confidence
        })
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return cv_image, result


#############################################
# MTCNN Face Detection
def detect_faces_mtcnn(image):
    mtcnn = load_mtcnn_model()
    boxes, probs = mtcnn.detect(image, landmarks=False)
    result = []
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    if boxes is not None:
        for idx, (box, prob) in enumerate(zip(boxes, probs)):
            x1, y1, x2, y2 = [int(v) for v in box]
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
            draw.text((x1, y1 - 15), f"Face {idx+1}", fill="blue", fontsize=12)
            result.append({
                "Face ID": f"Face {idx+1}",
                "X": x1,
                "Y": y1,
                "W": x2 - x1,
                "H": y2 - y1,
                "Confidence": f"{prob:.2f}"
            })
    return draw_image, result


#############################################
# Dlib Face Detection
def detect_faces_dlib(image):
    dlib_detector = load_dlib_detector()
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    dets, scores, _ = dlib_detector.run(cv_image, 1, -1)

    result = []
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    for idx, (d, score) in enumerate(zip(dets, scores)):
        if score > 0.0:
            x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1 - 15), f"Face {idx+1}", fill="red")
            result.append({
                "Face ID": f"Face {idx+1}",
                "X": x1,
                "Y": y1,
                "W": x2 - x1,
                "H": y2 - y1,
                "Confidence": f"{score:.2f}"
            })
    return draw_image, result



# #############################################
# Face Cropping Function
def get_face_crops(image, boxes):
    faces = []
    for box in boxes:
        x, y, w, h = box["X"], box["Y"], box["W"], box["H"]
        cropped_face = image.crop((x, y, x + w, y + h))
        cropped_face = ImageOps.fit(cropped_face, (80, 80))
        faces.append(cropped_face)
    return faces


#############################################
# Main App
def skin_tone_bias_face_detection():
    st.header("👤 Skin Tone Bias in Face Detection")
    st.markdown(
        """
        This demo shows potential **skin tone bias** in different face detection models like **Dlib**, **MTCNN**, and **OpenCV**.
        For instance, a group image with diverse skin tones might lead to different detection results across models.
        Try uploading group images with a mix of skin tones to observe how well each model performs.
        """
    )

    model_choice = st.selectbox("Select a Face Detection Model", ["Dlib", "MTCNN", "OpenCV"])
    input_method = st.selectbox("Select Input Method", ["Default Images", "Upload Image", "Use Image URL"])
    image = None

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload a group image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

    elif input_method == "Use Image URL":
        image_url = st.text_input("Paste an image URL")
        if image_url:
            try:
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            except Exception:
                st.error("Couldn't load image from the provided URL.")

    elif input_method == "Default Images":
        default_path = "default_images/skin_tone_bias"
        if os.path.exists(default_path):
            default_images = sorted([f for f in os.listdir(default_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            if default_images:
                selected = st.selectbox("Choose a default image", default_images)
                image = Image.open(os.path.join(default_path, selected)).convert("RGB")
            else:
                st.warning("No images found in 'default_images/skin_tone_bias'.")
        else:
            st.warning("Folder 'default_images/skin_tone_bias' does not exist.")

    if image is not None:
        st.image(image, caption="Input Image", width=400)

    if st.button("🔍 Detect Faces"):
        if image is None:
            st.warning("⚠️ Please provide an image before detection.")
        else:
            if model_choice not in ["OpenCV", "MTCNN", "Dlib"]:
                st.error("⚠️ Please select a valid model.")
            else:
                with st.spinner(f"Detecting faces using {model_choice}..."):
                    if model_choice == "OpenCV":
                        draw_image, result = detect_faces_opencv(image)
                    elif model_choice == "MTCNN":
                        draw_image, result = detect_faces_mtcnn(image)
                    elif model_choice == "Dlib":
                        draw_image, result = detect_faces_dlib(image)

                    if result:
                        st.success(f"✅ Detected {len(result)} face(s) with {model_choice}")
                        st.image(draw_image, caption=f"{model_choice} Detection Output", use_container_width=True)
                        # st.subheader("📊 Detection Details")
                        # df = pd.DataFrame(result)
                        # df.index += 1  # Start index from 1
                        # st.table(df)
                        # Get cropped face images

                        face_images = get_face_crops(image, result)
                        # Create a DataFrame with Face ID, Confidence, and face image
                        df_data = [
                            {
                                "Face": face_images[i],
                                "Face ID": result[i]["Face ID"],
                                "Confidence": result[i]["Confidence"]
                            }
                            for i in range(len(result))
                        ]

                        # Display cropped faces in a grid: 2 per row, with confidence below each
                        st.markdown("### 👤 Cropped Face Previews")

                        num_faces = len(df_data)
                        cols_per_row = 3

                        for i in range(0, num_faces, cols_per_row):
                            cols = st.columns(cols_per_row, gap="large")  # Add space between columns
                            for j in range(cols_per_row):
                                if i + j < num_faces:
                                    face_data = df_data[i + j]
                                    with cols[j]:
                                        st.image(face_data["Face"], use_container_width=True)
                                        st.markdown(
                                            f"<div style='text-align: center;'>"
                                            f"<b>{face_data['Face ID']}</b><br>"
                                            f"Confidence: <code>{face_data['Confidence']}</code>"
                                            f"</div>",
                                            unsafe_allow_html=True
                                        )
                            st.markdown("<br>", unsafe_allow_html=True)  # Add space between rows
                    else:
                        st.warning(f"⚠️ No faces detected by {model_choice}. Try a different model or image.")


if __name__ == "__main__":
    skin_tone_bias_face_detection()
