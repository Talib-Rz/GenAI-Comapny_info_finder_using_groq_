import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

# Set Groq API Key
# os.environ["GROQ_API_KEY"] = "your_key"

st.title("Company Info Finder (Powered by Groq-LLM)")
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

llm = ChatGroq(temperature=0, groq_api_key="your_key", model_name="mixtral-8x7b-32768")


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
