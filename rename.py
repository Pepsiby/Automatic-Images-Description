import os
import pyexiv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


model_name = "Salesforce/blip-image-captioning-base"
tokenizer = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

def generate_image_caption(img_path):
    image = Image.open(img_path)
    inputs = tokenizer(image, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return caption

def rename_and_update_metadata(file_path, caption):
    safe_caption = caption[:200]  
    new_file_name = f"{safe_caption}.jpg"  
    new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)


    os.rename(file_path, new_file_path)
    print(f"Renamed to '{new_file_name}'")


    with pyexiv2.Image(new_file_path) as img:
        img.modify_exif({'Exif.Image.ImageDescription': caption})
        img.modify_iptc({'Iptc.Application2.Caption': caption})
        img.modify_xmp({'Xmp.dc.description': caption})
        print(f"Added caption to metadata for '{new_file_name}'")


current_folder_path = os.getcwd()
image_files = [f for f in os.listdir(current_folder_path) if f.endswith('.jpg') or f.endswith('.png')]

for filename in image_files:
    img_path = os.path.join(current_folder_path, filename)
    caption = generate_image_caption(img_path)
    rename_and_update_metadata(img_path, caption)
