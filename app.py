import streamlit as st
# from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
# from PIL import Image
# import cv2
# import requests
# import torch
# import numpy as np

# Importing custom modules
from BiasedVisionModel import clothing_bias_scene_classification
from BiasedVisionModel import makeup_hair_gender_detection
from BiasedVisionModel import skin_tone_bias_face_detection
from BiasedVisionModel import gender_bias_occupation_detection

def main():
    st.title("Bias in Vision Pipelines")
    task = st.radio("Select a Task", [
        "Gender Bias in Occupation Detection",
        "Skin Tone Bias in Face Detection",
        "Makeup / Hair Bias in Gender Detection",
        "Clothing Bias in Scene Classification"
    ], index=0)

    if task == "Gender Bias in Occupation Detection":
        gender_bias_occupation_detection()
    elif task == "Skin Tone Bias in Face Detection":
        skin_tone_bias_face_detection()
    elif task == "Makeup / Hair Bias in Gender Detection":
        makeup_hair_gender_detection()
    elif task == "Clothing Bias in Scene Classification":
        clothing_bias_scene_classification()

if __name__ == "__main__":
    main()
