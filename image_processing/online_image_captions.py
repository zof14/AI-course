import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration

url = "https://en.wikipedia.org/wiki/IBM"
response = requests.get(url)
soup = BeautifulSoup(response.text,'html.parser')

#finding images
imgs = soup.find_all('img')
#load processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
with open('image_captions.txt','w') as captions:
    for image in imgs:
        image_url = image.get('src')
        if 'svg' in image_url or '1x1' in image_url:
            continue

        if image_url.startswith('//'):
            image_url = 'https' + image_url
        elif not image_url.startswith('http://') and not image_url.startswith('https://'):
            continue
        try:
            response = requests.get(image_url)
            raw_image = Image.open(BytesIO(response.content))
            if raw_image.size[0] * raw_image.size[1] < 400:  # Skip very small images
                continue
            inputs = processor(raw_image,retur_tensors="pt")
            out = model.generate(**inputs,max_new_tokens=50)
            caption = processor.decode(out[0],skip_special_tokens=True)
            captions.write(f"{image_url}: {caption}\n")
        except Exception as e:
            print(f"error processing image {image_url}: {e}")
            continue


