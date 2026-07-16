import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

# 1. Model Load
print("Checking Model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def verify():
    img_path = "test.jpg" # Check karein ki ye file 'services' folder mein hai
    complaint = "a photo of a pothole"

    if not os.path.exists(img_path):
        print(f"ERROR: '{img_path}' file nahi mili! Please image ko 'services' folder mein rakhein.")
        return

    try:
        image = Image.open(img_path)
        print(f"Analyzing Image: {img_path}...")
        
        # CLIP Comparison
        inputs = processor(text=[complaint, "a random object"], images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        
        probs = outputs.logits_per_image.softmax(dim=1)
        score = probs[0][0].item() * 100
        
        print("\n" + "="*30)
        print(f"RESULT FOR: {complaint}")
        print(f"MATCH SCORE: {score:.2f}%")
        print("="*30)
        
        if score > 60:
            print("STATUS: ✅ VERIFIED (Valid Complaint Image)")
        else:
            print("STATUS: ❌ REJECTED (Image mismatch)")

    except Exception as e:
        print(f"An error occurred: {e}")

# Yeh line sabse zaroori hai testing ke liye
if __name__ == "__main__":
    verify()