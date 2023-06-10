import os
from typing import Any, Dict, List
from langchain.embeddings import VertexAIEmbeddings
from langchain.chat_models import ChatVertexAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv

load_dotenv()

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = VertexAIEmbeddings()  # Dimention 768

    vectorstore = Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=os.environ["PINECONE_INDEX_NAME"],
    )
    chat = ChatVertexAI(
        verbose=True,
        temperature=0
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=vectorstore.as_retriever(), return_source_documents=True, verbose=True
    )
    return qa({"question": query, "chat_history": chat_history})

# Run locally
if __name__ == "__main__":
    print(run_llm(query="What is this application about?"))