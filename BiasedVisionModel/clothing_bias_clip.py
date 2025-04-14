import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import os
import torch
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import matplotlib.pyplot as plt

# Load model
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
    return model, processor


def clothing_bias_scene_classification():

    model, processor = load_clip_model()

    st.title("👖 Clothing Bias in Scene Classification 👗")
    st.markdown("""
    This application explores biases in scene classification models related to clothing attributes.
    It leverages the CLIP model to analyze and highlight these biases.
    """)

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
            except:
                st.error("Couldn't load image from the provided URL.")
    elif input_method == "Default Images":
        default_choice = st.selectbox("Choose from default images", sorted([f for f in os.listdir("default_images/clothing_bias") if f.lower().endswith((".jpg", ".png", ".jpeg"))]), index=0)
        image = Image.open(os.path.join("default_images/clothing_bias", default_choice)).convert("RGB")

    # Show the image only if already loaded
    if image is not None:
        st.image(image, caption="Input Image", width=250)

    # Prompt input
    st.subheader("📝 Candidate Scene Labels")
    default_prompts = ["a business executive", "a festival participant"]
    prompts_text = st.text_area("Enter one label per line:", "\n".join(default_prompts))
    labels = [label.strip() for label in prompts_text.strip().split("\n") if label.strip()]

    # Process and classify
    if st.button("🔍 Analyze Image"):
        if image is None:
            st.warning("⚠️ Please upload an image, paste a URL, or choose a default image before analysis.")
        else:
            with st.spinner("Analyzing the image, please wait..."):
                inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)[0]

            # Show probabilities
            st.subheader("📊 Classification Probabilities")
            data = {"Label": labels, "Probability": probs.numpy()}
            df = pd.DataFrame(data)
            df.index += 1  # Start index from 1
            st.table(df)
            st.write("**Most likely label**:", labels[probs.argmax().item()])
            st.write("\n")

            # Bar plot
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(labels, probs.numpy(), color='skyblue')
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.set_title("Scene Classification")
            st.pyplot(fig)


if __name__ == "__main__":
    clothing_bias_scene_classification()
