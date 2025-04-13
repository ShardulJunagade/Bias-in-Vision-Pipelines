import streamlit as st
from PIL import Image
import requests
import os

def makeup_hair_bias_gender_detection():
    st.title("Makeup / Hair Bias in Gender Detection")
    model_choice = st.selectbox("Select Model", ["Amazon Rekognition", "OpenCV"], index=0)
    input_method = st.selectbox("Select Input Method", ["Upload Image", "Use Image URL"], index=0)

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
        else:
            st.write("No image uploaded. Using default image.")
            default_path = os.path.join("default_images", "man_with_long_hair.jpg")
            image = Image.open(default_path).convert('RGB')
    else:
        img_url = st.text_input("Enter image URL:")
        if img_url:
            image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        else:
            st.write("No URL provided. Using default image.")
            default_path = os.path.join("default_images", "man_with_long_hair.jpg")
            image = Image.open(default_path).convert('RGB')

    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write(f"Model {model_choice} support is under development.")

if __name__ == "__main__":
    makeup_hair_bias_gender_detection()
