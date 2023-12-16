import base64
from vertexai.preview.vision_models import Image, ImageGenerationModel


def generate_image(prompt, model_name="imagegeneration@005"):
    model = ImageGenerationModel.from_pretrained(model_name=model_name)
    images = model.generate_images(prompt=prompt, number_of_images=1)
    return images[0]
