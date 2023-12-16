from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from vertexai.preview.generative_models import GenerativeModel, Part


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


def generate_caption_gemini(gcs_url):
    model = GenerativeModel("gemini-pro-vision")
    print(gcs_url)
    image = Part.from_uri(gcs_url, mime_type="image/png")
    responses = model.generate_content(
        contents=[image, """Write a creative and extremely funny meme caption inspired by this image. The caption should not be longer than 10 words and must mention something that is present in the image."""],
        generation_config={
            "max_output_tokens": 2048,
            "temperature": 1,
            "top_p": 1,
            "top_k": 32,
        },
    )
    return responses.candidates[0].content.parts[0].text
