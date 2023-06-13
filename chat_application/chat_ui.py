import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from chat_backend import run_llm
from dotenv import load_dotenv
import os

load_dotenv()

def get_text(instruction: str = "You: "):
    input_text = st.text_input(instruction, "", key=f"input-{instruction}")
    return input_text


st.set_page_config(page_title="MemeAI Chatbot - AI powered chatbot")
with st.sidebar:
    st.title("💬 MemeAI Chat")
    st.markdown(
        """
    ## About
    This app is an LLM-powered chatbot built using:
    - [LangChain 🦜🔗](https://python.langchain.com/en/latest/index.html)
    - [Pinecone 🌲 Vectorestore](https://www.pinecone.io/)
    - [Palm2 LLM Model](https://ai.google/discover/palm2)   
    - [Streamlit](https://streamlit.io/)
    """
    )

if "generated" not in st.session_state:
    st.session_state["generated"] = ["I am CommitAssist, I'm here to answer any question you have about this application. Ask me anything!"]
if "past" not in st.session_state:
    st.session_state["past"] = ["Hi!"]
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


input_container = st.container()
colored_header(label="", description="", color_name="blue-30")
response_container = st.container()


with input_container:
    user_input = get_text(instruction="You: ")


## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        with st.spinner("Generating response..."):
            response = run_llm(
                query=user_input, chat_history=st.session_state["chat_history"]
            )
            st.session_state.past.append(user_input)
            st.session_state.generated.append(response["answer"])
            st.session_state["chat_history"].append((user_input, response["answer"]))

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))