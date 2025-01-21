import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser, JsonOutputParser, CommaSeparatedListOutputParser
from langchain.output_parsers import DatetimeOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough, chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from typing import Optional
from pydantic import BaseModel, Field
from sqlalchemy import create_engine 
from sqlalchemy.engine import Engine 
from dotenv import load_dotenv
load_dotenv('.env')

st.title("Your own Chatbot")
st.write("This is a simple chatbot that can help you with your queries. Try asking it anything!")

base_url = "http://localhost:11434"
model = "llama3.2:latest"       # model = "llama3.2:1b"
user_id = st.text_input("Enter your user ID:", "utkarsh_kant")

def get_session_history(session_id):     
    engine: Engine = create_engine("sqlite:///chatbot_history.db")    # Create a database engine     
    return SQLChatMessageHistory(session_id, connection=engine)    # Use the database engine instead of a connection string     


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if st.button("Start New Conversation"):
    st.session_state.chat_history = []
    history = get_session_history(user_id)
    history.clear()

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

llm = ChatOllama(base_url=base_url, model=model, temperature=0.1)
system = SystemMessagePromptTemplate.from_template("You are helpful assistant")
human = HumanMessagePromptTemplate.from_template("{input}")
messages = [system, MessagesPlaceholder(variable_name='history'), human]
prompt = ChatPromptTemplate(messages=messages)
chain = prompt | llm | StrOutputParser()
runnable_with_history = RunnableWithMessageHistory(chain, get_session_history, 
                                                   input_messages_key='input', 
                                                   history_messages_key='history')

def chat_with_llm(session_id, input):
    # output = runnable_with_history.invoke(
    for output in runnable_with_history.stream(
                {'input': input},
                config= {'configurable': {'session_id': session_id}}
                ):
        yield output
    # return output

prompt = st.chat_input("What is up?")
# st.write(prompt)

if prompt:
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = st.write_stream(chat_with_llm(user_id, prompt))

    st.session_state.chat_history.append({'role': 'bot', 'content': response})
