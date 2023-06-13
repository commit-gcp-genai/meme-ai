# Meat and Learn Demo Applications
This repository contains two applications:

1. MemeAI: A Python Flask application that uses Google Cloud AI services to generate captions for images and create memes.
2. MemeAI Chat: A Streamlit chatbot application that uses Langchain and Vertex AI to chat with users and answer questions about the repository.

## Prerequisites
- Python 3.7 or higher
- Google Cloud account with the necessary APIs enabled
- Google Cloud SDK installed and configured
- Flask and other required Python packages (install using pip install -r requirements.txt)

## MemeAI Application
The MemeAI application is a Python Flask application that uses Google Cloud AI services to generate captions for images and create memes. The application uses the following Google Cloud services:

- [Vision API](https://cloud.google.com/vision) to detect objects in images
- [Language Models](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview) to generate captions for images
- [Translate API](https://cloud.google.com/translate) to translate captions to different languages
- [Text-to-Speech API](https://cloud.google.com/text-to-speech) to convert captions to speech
- [Cloud Storage](https://cloud.google.com/storage) to store images and audio files
- [Datastore](https://cloud.google.com/datastore) to store image metadata

### How does it work?

1. Image Upload and Processing
- Upload an image: Click on the "Choose File" button on the homepage and select an image file from your local machine. Click the "Upload" button to submit the image to the application.


2. Caption Generation
- Generate a caption using Vision API labels and a Language Model (LLM): After uploading an image, click on the "Generate Caption" button. The application will send the image to the Vision API to retrieve labels, and then use an LLM model to generate a caption based on the labels.

The LLM caption generation is performed using the following code:

```python
from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def generate_caption(labels):
    """
    Generates a caption for a given set of labels
    :param labels: A list of labels
    :return: A caption
    """

    llm = VertexAI(temperature=1, max_output_tokens=20)

    prompt = PromptTemplate(
        input_variables=["labels"],
        template="""You are an extremely funny meme caption generator. 
You will receive a list of items found within an image and generate a caption that relates to the items that you are given.
The caption should not be longer than 10 words.

items: {labels}
caption:""",
    )

    print(prompt.format(labels=labels))

    chain = LLMChain(llm=llm, prompt=prompt)

    return chain.run(labels=labels)
```
The generate_caption function takes a list of labels as input, creates an LLMChain with a specified LLM model and prompt, and generates a caption based on the provided labels.

3. Meme Generation
- Add a caption to an image: After uploading an image, enter a caption in the provided text field and click the "Meme" button. The application will use the caption and the original image to generate a meme image.

4. Caption Translation
- Translate a caption to a different language: After uploading an image and generating a caption, select a target language from the dropdown menu and click the "Translate" button. The application will use the Google Translate API to translate the caption to the selected language.

5. Text-to-Speech Conversion
- Convert a caption to speech: After uploading an image and generating a caption, click the "Text-to-Speech" button. The application will use the Google Text-to-Speech API to convert the caption to an MP3 audio file.

6. Image analysis
- View image metadata: After uploading an image, click the "Analyze Image" button. The application will use the Cloud Vision API to detect objects in the image, display the detected objects, and store the image metadata in a Datastore entity.

7. Image Generation
- Generate an image based on a text prompt: Write a prompt then click on the "Generate" button. The application will use the Stable Diffusion model to generate an image based on the prompt, upload the image to Cloud Storage, and create a Datastore entity with the image metadata.

## Chatbot Application
This application demonstrates an AI-powered Chatbot that uses an index of vectors and LLMs (Language Models) to provide conversational responses related to the repository.

The Langchain library is used to facilitate working with LLMs as well as creating and querying vector indexes by using it's numerous helper functions, modules and wrappers. The library also provides a convenient interface for working with the Vertex AI API. This is only an example of what can be achieved with Langchain, and the library can be used to create a variety of different applications.

### How does it work?

1. To enable the AI Chatbot to provide relevant responses, the repository data needs to be ingested and indexed. The ingest.py script facilitates this process by using Langchain to ingest the repository data and create an index of vectors. The script also uses the Vertex AI API to create an LLM model.

2. Once the index has been prepared, the chat_backend.py contains the backend logic for the chatbot. It performs the following tasks:
- Initializes the necessary dependencies, such as embeddings and vector stores.
- Defines a function run_llm(query, chat_history) that takes a user query and chat history as input and generates a response using the LLM and vector index.

3. The conversational chain is then used within the chat_ui.py script to create a Streamlit application that allows users to chat with the AI Chatbot.

