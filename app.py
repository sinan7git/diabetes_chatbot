import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate   
from langchain_groq import ChatGroq

# Load environment variables (to get API keys, etc.)
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure the Google API key
GoogleGenerativeAIEmbeddings.api_key = api_key

# Function to load and split text from a Diabetes-related PDF
def get_file_text(pdf_path='data/diabetes-information.pdf'):
    text = ""
    if not os.path.exists(pdf_path):
        st.error(f"PDF file not found at {pdf_path}. Please provide a valid file.")
        return None

    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Ensure the FAISS index exists, or create it if missing
def load_or_create_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_path = "faiss_index/index.faiss"

    if not os.path.exists(faiss_path):
        text = get_file_text()
        if text is None:
            return None  # No text available
        text_chunks = get_text_chunks(text)
        return get_vector_store(text_chunks)

    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Create a conversational chain for answering diabetes-related questions
def get_conversational_chain(db, question):
    retriever = db.as_retriever()
    model = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key='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')

    template = """You are a helpful AI assistant with a broad knowledge in the medical domain, 
    especially in diabetes. The user asks a question: {question}. Provide an answer based on the context. 
    If you do not know the answer, say "I do not have knowledge." Do not fabricate information. {context}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    chain = setup_and_retrieval | prompt | model | output_parser
    response = chain.invoke(question)    
    return response

# Main function
def main():
    st.set_page_config("Diabetes Chatbot")
    st.header("Diabetes Trained Chatbot 🌟")

    # Initialize vector store
    db = load_or_create_vector_store()
    if db is None:
        st.error("Failed to initialize the vector store. Please check the logs.")
        return

    # Example questions
    import random
    ques = [
        'Can type 1 diabetes be cured?',
        'What are the signs and symptoms of diabetes?',
        'What is the history of diabetes?'
    ]
    
    selected_ques = random.choice(ques)
    if st.button('Generate Random Question'):
        st.write(selected_ques)
        user_question = selected_ques

        with st.spinner('Thinking...'):
            if user_question:
                response = get_conversational_chain(db, user_question)
                st.write(response)
    
    user_question = st.text_input("Ask a Question about Diabetes")
    if st.button("Submit"):
        with st.spinner('Thinking...'):
            if user_question:
                response = get_conversational_chain(db, user_question)
                st.write(response)

if __name__ == "__main__":
    main()
