import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

# For OpenCV-based model
import cv2
import numpy as np

# For CLIP
from transformers import CLIPProcessor, CLIPModel

#############################################
# CLIP Model Loader (cached)
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # Note: using use_fast=False sometimes gives more consistent tokenization
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
    return model, processor


#############################################
# CLIP Gender Detection Function
def analyze_with_clip(image):
    model, processor = load_clip_model()
    candidate_labels = ["male", "female"]
    inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
    result = {label: float(prob) for label, prob in zip(candidate_labels, probs)}
    most_likely = max(result, key=result.get)
    return result, most_likely


#############################################
# OpenCV Gender Classifier
def analyze_with_opencv(image):
    # Path to the Caffe model files.
    gender_proto = "models/gender_deploy.prototxt"
    gender_model = "models/gender_net.caffemodel"

    # Check for model file existence
    if not os.path.exists(gender_proto) or not os.path.exists(gender_model):
        return "OpenCV gender model files not found. Please ensure they exist in the 'models' folder."

    net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)
    # The expected labels (this ordering may vary depending on the trained model)
    gender_list = ["Male", "Female"]

    # Convert PIL Image to OpenCV image (BGR)
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Use Haar cascades for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    results = []
    for (x, y, w, h) in faces:
        face_img = cv_image[y:y+h, x:x+w]
        # Preprocess face region: resize to the network input size (227,227)
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                     (78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False)
        net.setInput(blob)
        preds = net.forward()
        results = {label: float(prob) for label, prob in zip(gender_list, preds[0])}
        # Get the label with the highest probability
        most_likely = max(results, key=results.get)
        # print(results)
        return results, most_likely



#############################################
# Main Task: Makeup/Hair Bias in Gender Detection
def makeup_hair_gender_detection():
    st.title("💄 Makeup / Hair Bias in Gender Detection 💇‍♂️")
    st.markdown(
        "This demo shows potential biases in gender detection. For instance, a woman with short hair might be misclassified as **male**, "
        "and a man with long hair might be misclassified as **female**. Use an image from your computer, a URL, or select one of the pre‐curated examples."
    )

    model_choice = st.selectbox("Select a Model", ["OpenCV", "CLIP"])
    input_method = st.selectbox("Select Input Method", ["Default Images", "Upload Image", "Use Image URL"], index=0)
    image = None

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload your own image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
    elif input_method == "Use Image URL":
        image_url = st.text_input("Paste an image URL")
        if image_url:
            try:
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as e:
                st.error("Couldn't load image from the provided URL.")
    elif input_method == "Default Images":
        default_images = sorted([f for f in os.listdir("default_images/makeup_hair_bias") if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        if default_images:
            default_choice = st.selectbox("Choose from default images", default_images, index=0)
            image = Image.open(os.path.join("default_images/makeup_hair_bias", default_choice)).convert("RGB")
        else:
            st.warning("No default images found in 'default_images/makeup_hair_bias' folder.")

    # Display the chosen image
    if image is not None:
        st.image(image, caption="Input Image", width=250)

    # Run the selected analysis when the user clicks the button
    if st.button("🔍 Analyze Image"):
        if image is None:
            st.warning("⚠️ Please provide an image before analysis.")
        else:
            with st.spinner(f"Analyzing with {model_choice}..."):
                if model_choice == "OpenCV":
                    result, most_likely = analyze_with_opencv(image)
                elif model_choice == "CLIP":
                    result, most_likely = analyze_with_clip(image)

            st.subheader(f"📊 {model_choice} Gender Detection Results")
            if isinstance(result, dict):  # For CLIP or OpenCV single face
                st.write("Probabilities:")
                df = pd.DataFrame(list(result.items()), columns=["Label", "Probability"])
                df.index += 1  # Start index from 1
                st.table(df)
                st.write("**Most likely label:**", most_likely)
                st.write("\n")
                # Bar plot
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(list(result.keys()), list(result.values()), color='skyblue' if model_choice == "CLIP" else 'lightcoral')
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability")
                ax.set_title("Gender Classification")
                st.pyplot(fig)
            else:  # For OpenCV when no faces or multiple faces
                st.warning(result)

if __name__ == "__main__":
    makeup_hair_gender_detection()
