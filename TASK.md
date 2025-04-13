# Task: Bias in Vision Pipelines

## 🎯 1. Gender Bias in Occupation Detection
#### 🔍 Demo:
- Use an image classifier or captioning model on:
  - **A woman in a lab** → Predicted as "nurse"
  - **A man in the same lab** → Predicted as "scientist" or "engineer"

#### 🔧 Tools:
- **BLIP** (Salesforce image captioning model)
- **CLIP** (for zero-shot classification)

#### ✅ Code Snippet (BLIP):
```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

img_url = "https://example.com/woman_in_lab.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)
print("Caption:", processor.decode(out[0], skip_special_tokens=True))
```

#### 🧠 What to show:
- Same setting, different person — observe caption drift.

---

## 🎯 2. Skin Tone Bias in Face Detection
#### 🔍 Demo:
- Load multiple faces with varying skin tones.
- Run a face detector like **dlib**, **MTCNN**, or **OpenCV**.
- Show lower detection rate or confidence for darker skin.

#### ✅ Code Snippet (OpenCV):
```python
import cv2
image = cv2.imread("group_photo.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow("Detected Faces", image)
```

#### 🧠 Talking Point:
- "Most AI training datasets are skewed toward lighter skin tones."

---

## 🎯 3. Makeup / Hair Bias in Gender Detection
#### 🔍 Demo:
- Try **Amazon Rekognition** or **OpenCV gender classifier**:
  - **Woman with short hair** → Predicted as "male"
  - **Man with long hair** → Predicted as "female"
- Simulate this with image uploads or use pre-curated examples.

---

### 🎯 4. Clothing Bias in Scene Classification
#### Demo:
- Use **CLIP** or image tagging APIs:
  - **Sari** → Tagged as "Cultural" or "Festival"
  - **Suit** → Tagged as "Business"

#### Code Snippet (CLIP):
```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("woman_in_sari.jpg")
texts = ["a business executive", "a festival participant", "a teacher"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)
print("Probabilities:", probs)
```

#### Talking Point:
- 🧠 Even clothing carries latent cultural biases in AI.