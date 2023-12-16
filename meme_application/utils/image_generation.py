import base64
from vertexai.preview.vision_models import Image, ImageGenerationModel


def generate_image(prompt):

    model = ImageGenerationModel.from_pretrained("imagegeneration@005")
    images = model.generate_images(
        prompt=prompt,
        # Optional:
        number_of_images=1,
        seed=1
    )
    return images[0]


# Run locally
if __name__ == "__main__":
    image_request = generate_image(
        project="860550299524",
        endpoint="8743606735144484864",
        location="europe-west1",
        instances=[{"prompt": "A cat ridind a bycicle on the moon"}],
    )
    image_base64 = image_request.predictions[0]
    # Save the image to a file
    with open("/tmp/image.png", "wb") as fh:
        fh.write(base64.b64decode(image_base64))
