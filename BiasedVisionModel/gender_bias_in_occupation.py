import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from PIL import Image
import requests
import os
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource
def load_clip_model():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return processor, model

def gender_bias_occupation_detection():
    st.title("👩‍🔬 Gender Bias in Occupation Detection 👨‍🔬")
    st.markdown(
        "This demo highlights potential biases in occupation-related image tagging. "
        "For example, a woman in a lab might be labeled as a **nurse**, while a man in the same lab might be labeled as a **scientist**."
    )

    model_choice = st.selectbox("Select Model", ["CLIP", "BLIP"], index=0)
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
        default_images = sorted([f for f in os.listdir("default_images/occupation_bias") if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        if default_images:
            default_choice = st.selectbox("Choose from default images", default_images, index=0)
            image = Image.open(os.path.join("default_images/occupation_bias", default_choice)).convert("RGB")
        else:
            st.warning("No default images found in 'default_images/occupation_bias' folder.")

    if image is not None:
        st.image(image, caption="Input Image", width=250)

        if st.button("🔍 Analyze Image"):
            with st.spinner(f"Analyzing with {model_choice}..."):
                if model_choice == "BLIP":
                    processor, model = load_blip_model()
                    inputs = processor(image, return_tensors="pt")
                    out = model.generate(**inputs)
                    caption = processor.decode(out[0], skip_special_tokens=True)

                    st.subheader("📋 Caption")
                    st.write(f"**{caption}**")

                elif model_choice == "CLIP":
                    processor, model = load_clip_model()
                    texts = ["a nurse", "a doctor", "a scientist", "an engineer"]
                    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
                    outputs = model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]

                    st.subheader("📊 Classification Probabilities")
                    data = {"Label": texts, "Probability": probs}
                    df = pd.DataFrame(data)
                    df.index += 1  # Start index from 1
                    st.table(df)
                    st.write("**Most likely label**:", texts[probs.argmax()])

                    # Bar plot
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.barh(texts, probs, color='skyblue')
                    ax.set_xlim(0, 1)
                    ax.set_xlabel("Probability")
                    ax.set_title("Occupation Classification")
                    st.pyplot(fig)
    else:
        st.warning("⚠️ Please provide an image before analysis.")

if __name__ == "__main__":
    gender_bias_occupation_detection()
