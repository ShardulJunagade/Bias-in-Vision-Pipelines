# Bias in Vision Pipelines

This repository explores biases in computer vision pipelines, focusing on how AI models can exhibit unintended biases based on factors like gender, clothing, skin tone, and more. The project includes demos and code to highlight these biases and foster awareness.

> **Note for Deployment**: If you plan to deploy this project to Hugging Face Spaces, make sure to use this [specialized requirements file](./requirements_for_huggingface.txt) instead of the default `requirements.txt`. This ensures compatibility with the Hugging Face environment and avoids potential dependency issues.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Demos](#demos)
- [Contributing](#contributing)
- [License](#license)

## Introduction
AI models are increasingly used in critical applications, but they can exhibit biases that reflect societal stereotypes or dataset imbalances. This project demonstrates such biases in vision pipelines and provides tools to analyze and understand them.

## Features
- **Skin Tone Bias in Face Detection**: Compare face detection performance across diverse skin tones using OpenCV, MTCNN, and Dlib.
- **Makeup/Hair Bias in Gender Detection**: Analyze how hairstyles and makeup influence gender classification.
- **Clothing Bias in Scene Classification**: Explore how clothing types affect scene tagging using CLIP.
- **Gender Bias in Occupation Detection**: Investigate biases in occupation-related image tagging.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/ShardulJunagade/bias-in-vision-pipelines.git
   cd bias-in-vision-pipelines
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have the required models and default images in the appropriate directories.

## Usage
Run the Streamlit app to explore the demos interactively:
```bash
streamlit run app.py
```

## Demos
### Skin Tone Bias in Face Detection
- Upload or select default group images with diverse skin tones.
- Compare face detection results across OpenCV, MTCNN, and Dlib.

### Makeup/Hair Bias in Gender Detection
- Test gender classification models with images of individuals with varying hairstyles and makeup.

### Clothing Bias in Scene Classification
- Use CLIP to analyze how clothing types (e.g., sari, suit) influence scene tagging.

### Gender Bias in Occupation Detection
- Examine how gender influences occupation-related image tagging.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
