import os
import streamlit as st
from keys import openai_key

from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"] = openai_key


st.title("Company Info Finder")
input_text = st.text_input("Search for the company you want!")


cname_memory = ConversationBufferMemory(input_key="name", memory_key="chat_history")
cfounded_memory = ConversationBufferMemory(input_key="cname", memory_key="chat_history")
chistory_memory = ConversationBufferMemory(input_key="cfounded", memory_key="description_history")

first_input_prompt = PromptTemplate(
    input_variables=["name"],
    template="Tell me about the company {name}."
)

second_input_prompt = PromptTemplate(
    input_variables=["cname"],
    template="When was {cname} founded?"
)

third_input_prompt = PromptTemplate(
    input_variables=["cfounded"],
    template="Mention 5 big similar companies founded around {cfounded} in India."
)

llm = OpenAI(temperature=0.7)

chain1 = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key="cname", memory=cname_memory)
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key="cfounded", memory=cfounded_memory)
chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key="chistory", memory=chistory_memory)

parent_chain = SequentialChain(
    chains=[chain1, chain2, chain3],
    input_variables=["name"],
    output_variables=["cname", "cfounded", "chistory"],
    verbose=True
)

if input_text:
    response = parent_chain.invoke({"name": input_text}) 

    st.write("### Company Info:")
    st.write(response["cname"])
    
    st.write("### Founded In:")
    st.write(response["cfounded"])

    st.write("### Similar Companies:")
    st.write(response["chistory"])
