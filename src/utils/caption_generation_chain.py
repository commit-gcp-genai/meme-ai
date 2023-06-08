from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def generate_caption(labels):
    """
    Generates a caption for a given set of labels
    :param labels: A list of labels
    :return: A caption
    """

    llm = VertexAI(temperature=1)

    prompt = PromptTemplate(
        input_variables=["labels"],
        template="""You are a meme caption generator. You will receive a list of labels and generate a funny caption that relates to the labels found in the image.
                    labels: {labels}
                    caption:""",
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    return chain.run(labels=labels)

# Run locally
if __name__ == "__main__":
    print(generate_caption(["dog", "cat", "bird"]))
