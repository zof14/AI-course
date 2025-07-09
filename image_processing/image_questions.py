import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

img_path = 'dog.jpeg'
raw_image = Image.open(img_path).convert('RGB')

question = "What animal is on the image?"

inputs = processor(raw_image, question, return_tensors="pt")

out = model.generate(**inputs)

answer = processor.decode(out[0], skip_special_tokens=True)
print(f"Answer: {answer}")