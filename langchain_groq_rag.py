import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import time
from dotenv import load_dotenv

from langchain_community.chat_models import ChatLiteLLM,ChatOllama

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI

load_dotenv()  #

#groq_api_key = os.environ['GROQ_API_KEY']


if "vector" not in st.session_state:

    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="OrdalieTech/Solon-embeddings-large-0.1")
    #st.session_state.embeddings = OllamaEmbeddings()

    st.session_state.loader = WebBaseLoader("https://cloud-pi-native.fr/agreement/faq.html")
#https://en.wikipedia.org/wiki/Special:Random")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n"],
#, " ", ""],
        length_function=len)
    st.session_state.documents = st.session_state.text_splitter.split_documents( st.session_state.docs)
    st.session_state.vector = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)

st.title("RAG démonstrateur")

st.write("""Voici le template créé pour que Mixtral réponde en français:
```
<contexte>
{context}
</contexte>

D'après le contexte répondez en français à cette requête: {input}
```
""")
#llm = ChatGroq(
#            groq_api_key=groq_api_key, 
#            model_name='mixtral-8x7b-32768'
#    )
llm = ChatOpenAI(model_name="mixtral",openai_api_key="EMPTY",openai_api_base="https://example.com/v1",temperature=.2)

prompt = ChatPromptTemplate.from_template("""
<contexte>
{context}
</contexte>

D'après le contexte répondez en français à cette requête: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = st.session_state.vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Entrer votre requête.")


# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print(f"Response time: {time.process_time() - start}")

    st.markdown(response["answer"])

    # With a streamlit expander
    with st.expander("Détails"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(f"Source Document # {i+1} : {doc.metadata['source'].split('/')[-1]}")
            st.write(doc.page_content)
            st.write("--------------------------------")
