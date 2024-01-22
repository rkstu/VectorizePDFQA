import streamlit as st
import numpy as np
import pandas as pd

import openai
import langchain
import pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
import time
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI

from helper_functions import (read_doc,
                              chunk_data,
                              retrieve_answer,
                              retrieve_query)

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]
PINECONE_ENVIRONMENT = st.secrets["PINECONE_ENVIRONMENT"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

def user_input(user_question):
    embeddings=OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    # print("Embeddings: ", embeddings)

    ## Vector search DB in  Pinecone
    pinecone.init(
      api_key=PINECONE_API_KEY,
      environment=PINECONE_ENVIRONMENT
    )
    index_name=PINECONE_INDEX_NAME
    time.sleep(2)

    llm = OpenAI(model_name="davinci-002", temperature=0.5)
    chain = load_qa_chain(llm, chain_type="stuff")

    # query = "Tell me something about AI?"
    index = Pinecone.from_documents("", embeddings, index_name=index_name)
    doc_search = index.similarity_search(user_question, k=2)
    answer=chain.run(input_documents=doc_search, question=user_question)
    answer = str(answer).replace('\n', ' ')


    # print(response)
    st.write("Reply: ", answer)





def main():
    st.set_page_config("Chat PDF")
    st.header("Personalized Search Engine Interface")
    st.markdown(""" Application utilizes open-source Language Model (LLM) technology, specifically leveraging models such as
        openAI's, Langchain, and Pinecone. It employs Fast and Fresh Vector Search using Pinecone as a cloud-based
        vector database. This application can generate personalized search results for any document,
        with the current [document](https://github.com/rkstu/Personalized-Search-Engine-Interface-and-Chatbot/tree/main/documents) focusing on topics related to AI and ML. Feel free to ask any questio related to the document.""")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)




if __name__ == "__main__":
    main()