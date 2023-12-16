import logging, os, random
from PIL import Image, ExifTags
from flask import Flask, redirect, render_template, request
from google.cloud import vision
import google.cloud.translate_v2 as translate
import google.cloud.texttospeech as tts
from utils.datastore_operations import DatastoreManager
from utils.memefy import Meme
from utils.caption_generation_chain import generate_caption
from utils.helpers import StorageHelpers
from utils.image_generation import generate_image
from google.cloud import aiplatform
import tempfile


# Define variables
CLOUD_STORAGE_BUCKET = os.environ.get("CLOUD_STORAGE_BUCKET")
IMAGE_GENERATION_PROJECT = os.environ.get("IMAGE_GENERATION_PROJECT")
IMAGE_GENERATION_LOCATION = os.environ.get("IMAGE_GENERATION_LOCATION")

# Initialize clients
datastore_manager = DatastoreManager()
vision_client = vision.ImageAnnotatorClient()
tts_client = tts.TextToSpeechClient()
translate_client = translate.Client()
helpers = StorageHelpers()

# Initialize app
app = Flask(__name__)
aiplatform.init(project=IMAGE_GENERATION_PROJECT, location=IMAGE_GENERATION_LOCATION)

# Render HTML template
@app.route("/")
def homepage():
    # Get all entities from Datastore
    image_entities = datastore_manager.query_entities("Memes")
    # Return a Jinja2 HTML template and pass in image_entities as a parameter.
    return render_template(
        "homepage.html", image_entities=image_entities, form_error=False
    )

# Submit image to Google Cloud Storage
@app.route("/upload_photo", methods=["GET", "POST"])
def upload_photo():
    photo = request.files["file"]
    # Open the image and check for its orientation tag
    image = Image.open(photo)
    orientation = None
    if hasattr(image, "_getexif") and image._getexif() is not None:
        for key, value in image._getexif().items():
            if key in ExifTags.TAGS and ExifTags.TAGS[key] == "Orientation":
                orientation = value
                break

    # Rotate the image based on the orientation tag
    if orientation == 3:
        image = image.transpose(Image.ROTATE_180)
    elif orientation == 6:
        image = image.transpose(Image.ROTATE_270)
    elif orientation == 8:
        image = image.transpose(Image.ROTATE_90)

    # Resize the image to a square of size 300x300 pixels
    image.thumbnail((300, 300))

    # Create a new square image with a white background
    new_image = Image.new("RGB", (300, 300), (255, 255, 255))
    new_image.paste(image, ((300 - image.width) // 2, (300 - image.height) // 2))

    # Save the resized and squared image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_image:
        new_image.save(temp_image.name, format="JPEG")

        # Read the resized image from the temporary file
        with open(temp_image.name, "rb") as f:
            resized_image = f.read()

        # Upload the resized image to Google Cloud Storage
        blob = helpers.upload_asset_to_bucket(
            resized_image, photo.filename, content_type="image/jpeg"
        )

    # Create a new entity in Datastore
    kind = "Memes"
    print("Creating Datastore entity")
    properties = {
        "original_image_blob": photo.filename,
        "original_image_public_url": blob.public_url,
        "processed": False,
    }
    datastore_manager.create_entity(kind, photo.filename, properties)
    print("Done")
    return redirect("/")


@app.route("/generate_image", methods=["GET", "POST"])
def generate_image_vertex():
    # Define a random uuid of 10 characters to use as the image name
    blob_name = "".join(random.choice("0123456789abcdef") for n in range(10))

    # Get prompt from form
    prompt = request.form["prompt"]
    model_name = request.form["model_name"]
    # generate the image using the image generation model and the provided prompt
    image = generate_image(prompt, model_name)
    image.save("/tmp/image.png")
    # Upload the image to Google Cloud Storage
    blob = helpers.upload_asset_to_bucket(
        "/tmp/image.png", blob_name, content_type="image/png"
    )

    # Create a new entity in Datastore
    kind = "Memes"
    properties = {
        "original_image_blob": blob_name,
        "original_image_public_url": blob.public_url,
        "processed": False,
    }
    datastore_manager.create_entity(kind, blob_name, properties)
    print("Done")
    return redirect("/")


# Image form button handler
@app.route("/process", methods=["GET", "POST"])
def process():
    blob_name = request.form["original_image_blob"]
    # Get image and check if it has a caption
    if request.form["action"] == "Update Caption":
        caption = (
            "Put a caption next time you press me"
            if len(request.form["caption"]) == 0
            else request.form["caption"]
        )
        memefy(blob_name, caption)
    # Generate a caption by using Vision API labels and an LLM model using a prompt template
    elif request.form["action"] == "Generate Caption":
        generate_image_caption(blob_name)
        return redirect("/")
    # Analyze the image using the Vision API
    elif request.form["action"] == "Analyze Image":
        analyze_image(blob_name)
        return redirect("/")
    elif request.form["action"] == "Delete":
        delete_image(blob_name)
        return redirect("/")
    entity = datastore_manager.get_entity("Memes", blob_name)
    # Return an error if the image selected has no caption
    if "caption" not in entity:
        image_entities = datastore_manager.query_entities("Memes")
        return render_template(
            "homepage.html",
            image_entities=image_entities,
            form_error="The image you selected does not have a caption. Please add a caption and try again.",
        )
    # Translate caption to target language using Google Translate API
    if request.form["action"] == "Translate":
        translate_target_lang = request.form["language"]
        translate_text(blob_name, translate_target_lang)
    # Convert caption to speech using Google Text-to-Speech API
    elif request.form["action"] == "Text-to-Speech":
        text_to_mp3(blob_name)

    # # Redirect to the home page.
    return redirect("/")


# Error Handling function
@app.errorhandler(500)
def server_error(e):
    logging.exception("An error occurred during a request.")
    return (
        """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(
            e
        ),
        500,
    )


# Use an LLM to generate a caption for an image given a list of lables from the Vision API
def generate_image_caption(blob_name):
    # Get the image entity from datastore
    entity = datastore_manager.get_entity("Memes", blob_name)
    # If entity has already labels, use them to generate a caption
    if "labels" in entity:
        labels_list = entity["labels"]
    # If entity does not have labels, analyze the image using the Vision API
    else:
        labels_list = analyze_image(blob_name)
    # Generate caption using LLM model and the list of labels
    caption = generate_caption(labels=labels_list)
    # Update image entity from datastore
    memefy(blob_name, caption)


# Uses PIL to write a caption on top of the image
def memefy(file_name, caption):
    # Download file
    original_image_blob = helpers.download_asset_from_bucket(file_name)
    # Detect caption language
    detected_lang = translate_client.detect_language(caption)
    # Generate meme
    meme = Meme(caption, helpers.asset_download_location, detected_lang["language"])
    img = meme.draw()
    if img.mode in ("RGBA", "P"):  # Without this the code can break sometimes
        img = img.convert("RGB")
    img.save("/tmp/captioned_image.jpg", optimize=True, quality=80)
    # Upload meme to bucket
    upload_image_blob = "processed/" + original_image_blob.name
    meme_blob = helpers.upload_asset_to_bucket(
        "/tmp/captioned_image.jpg", upload_image_blob, content_type="image/jpeg"
    )
    # Update image entity from datastore
    updates = {
        "processed_image_public_url": meme_blob.public_url,
        "caption": caption,
        "caption_language": detected_lang["language"],
        "processed": True,
    }
    datastore_manager.update_entity("Memes", file_name, updates)


# Translates text into the target language using Google cloud translate API
def translate_text(file_name, target_lang):
    # Get datastore entity for image
    entity = datastore_manager.get_entity("Memes", file_name)
    target = target_lang
    # Use Google translate API to translate caption into target language
    translated_text = translate_client.translate(
        values=entity["caption"], format_="text", target_language=target
    )
    # Generate meme with translated text
    memefy(file_name, translated_text["translatedText"])


# Choose a voice for the text-to-speech API based on the caption language
def get_voice(caption_language):
    if caption_language == "fr":
        voice = tts.VoiceSelectionParams(
            language_code="fr-FR", ssml_gender=tts.SsmlVoiceGender.FEMALE
        )
        return voice
    elif caption_language == "ru":
        voice = tts.VoiceSelectionParams(
            language_code="ru-RU", ssml_gender=tts.SsmlVoiceGender.MALE
        )
        return voice
    elif caption_language == "es":
        voice = tts.VoiceSelectionParams(
            language_code="es-ES", ssml_gender=tts.SsmlVoiceGender.FEMALE
        )
        return voice
    elif caption_language == "it":
        voice = tts.VoiceSelectionParams(
            language_code="it-IT", ssml_gender=tts.SsmlVoiceGender.MALE
        )
        return voice
    elif caption_language == "iw":
        voice = tts.VoiceSelectionParams(
            language_code="en-US", ssml_gender=tts.SsmlVoiceGender.NEUTRAL
        )
        return voice
    # In case the language doesn't match any of the above, chose an english voice
    else:
        voice = tts.VoiceSelectionParams(
            language_code="en-US", ssml_gender=tts.SsmlVoiceGender.NEUTRAL
        )
        return voice


# Converts text to an mp3 file using Text to Speech API
def text_to_mp3(blob_name):
    # Get the image caption from its Datastore entity
    entity = datastore_manager.get_entity("Memes", blob_name)
    # Prepare text to speech input
    synthesis_input = tts.SynthesisInput(text=entity["caption"])
    # Set file format for the response
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.MP3)
    # Get the voice parameters
    caption_language = entity["caption_language"]
    voice = get_voice(caption_language=caption_language)
    # Get mp3 file from Text to speech API
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    # Change file extension to mp3
    pre, ext = os.path.splitext(blob_name)
    upload_mp3_blob = "audio/" + pre + ".mp3"
    # Upload file
    mp3_blob = helpers.upload_asset_to_bucket(
        response.audio_content, upload_mp3_blob, content_type="audio/mp3"
    )
    updates = {
        "mp3_bucket_url": mp3_blob.public_url,
    }
    datastore_manager.update_entity("Memes", blob_name, updates)


# Analyze the image with the cloud vision API to get a list of labels
def analyze_image(blob_name):
    # Get the image entity from datastore
    entity = datastore_manager.get_entity("Memes", blob_name)
    # Get the image labels from the vision API
    image = vision.Image(
        source=vision.ImageSource(image_uri=entity["original_image_public_url"])
    )
    labels = vision_client.label_detection(image=image).label_annotations
    # Create a list of labels
    labels_list = [label.description for label in labels]
    entity["labels"] = labels_list
    updates = {
        "labels": labels_list,
    }
    datastore_manager.update_entity("Memes", blob_name, updates)
    return labels_list


# Delete the image from the bucket and the datastore
def delete_image(blob_name):
    # Delete image from datastore
    datastore_manager.delete_entity("Memes", blob_name)
    helpers.delete_asset_from_bucket(blob_name)
    print("Deleted image from bucket and datastore")


if __name__ == "__main__":
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host="0.0.0.0", port=8080, debug=True)
