import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from PIL import Image
import requests
import os

def gender_bias_occupation_detection():
    st.title("Gender Bias in Occupation Detection")
    model_choice = st.selectbox("Select Model", ["BLIP", "CLIP"], index=0)
    input_method = st.selectbox("Select Input Method", ["Upload Image", "Use Image URL"], index=0)

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
        else:
            st.write("No image uploaded. Using default image.")
            default_path = os.path.join("default_images/occupation_bias", "woman_in_lab.jpg")
            image = Image.open(default_path).convert('RGB')
    else:
        img_url = st.text_input("Enter image URL:")
        if img_url:
            image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        else:
            st.write("No URL provided. Using default image.")
            default_path = os.path.join("default_images/occupation_bias", "woman_in_lab.jpg")
            image = Image.open(default_path).convert('RGB')

    st.image(image, caption="Selected Image", use_column_width=True)

    if model_choice == "BLIP":
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        st.write("**Caption:**", caption)
    elif model_choice == "CLIP":
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        texts = ["a nurse", "a scientist", "an engineer"]
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]

        for text, prob in zip(texts, probs):
            st.write(f"**{text}:** {prob:.4f}")

if __name__ == "__main__":
    gender_bias_occupation_detection()
