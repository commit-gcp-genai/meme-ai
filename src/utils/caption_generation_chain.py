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

# Run locally
if __name__ == "__main__":
    print(generate_caption(["dog", "cat", "bird"]))
