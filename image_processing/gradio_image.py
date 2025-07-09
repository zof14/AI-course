from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import gradio as gr

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    inputs = processor(images=image,return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens = True)
    return caption
def caption_image(image):
    try:
        caption = generate_caption(image)
        return caption
    except Exception as e:
        return f"An error occurred: {str(e)}"
    
interface = gr.Interface(
    fn= caption_image,
    inputs = gr.Image(type = "pil"),
    outputs = "text",
    title = "Image Captioning with BLIP",
    description = "upload image to generate a caption")
interface.launch(server_name ="127.0.0.1", server_port = 7860)