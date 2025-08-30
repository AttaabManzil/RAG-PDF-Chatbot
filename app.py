from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

import gradio as gr

def warn(*args, **kwargs):
    pass 
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

def get_llm():
    try:
        model_id = 'mistralai/mixtral-8x7b-instruct-v01'
        params = {
            GenParams.MAX_NEW_TOKENS: 256,
            GenParams.TEMPERATURE: 0.3,
            GenParams.DECODING_METHOD: "greedy"
        }
        project_id = "skills-network"
        watsonx_llm = WatsonxLLM(
            model_id=model_id,
            params=params,
            project_id=project_id,
            url="https://us-south.ml.cloud.ibm.com"
        )
        return watsonx_llm
    except Exception as e:
        raise Exception(f"Error creating LLM: {str(e)}")

def document_loader(file):
    try:
        if file is None:
            raise Exception("No file provided")
        loader = PyPDFLoader(file)
        loaded_document = loader.load()
        if not loaded_document:
            raise Exception("PDF file appears to be empty or unreadable")
        return loaded_document
    except Exception as e:
        raise Exception(f"Error loading document: {str(e)}")

def text_splitter(data):
    try:
        if not data:
            raise Exception("No data provided for text splitting")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_documents(data)
        if not chunks:
            raise Exception("No chunks created from document")
        return chunks
    except Exception as e:
        raise Exception(f"Error splitting text: {str(e)}")

def watsonx_embedding():
    try:
        embed_params = {
            EmbedParams.TRUNCATE_INPUT_TOKENS: 512
        }
        watsonx_embedding = WatsonxEmbeddings(
            model_id="ibm/slate-125m-english-rtrvr",  # Fixed: removed 'bm/' prefix
            url="https://us-south.ml.cloud.ibm.com",
            params=embed_params,
            project_id="skills-network",
        )
        return watsonx_embedding
    except Exception as e:
        raise Exception(f"Error creating embedding model: {str(e)}")

def vector_database(chunks):
    try:
        embedding_model = watsonx_embedding()
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model
        )
        return vectordb
    except Exception as e:
        raise Exception(f"Error creating vector database: {str(e)}")

def retriever(file):
    try:
        splits = document_loader(file)
        chunks = text_splitter(splits)
        vectordb = vector_database(chunks)
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        return retriever
    except Exception as e:
        raise Exception(f"Error creating retriever: {str(e)}")

def retriever_qa(file, query):
    try:
        if not file:
            return "Error: Please upload a PDF file first."
        if not query.strip():
            return "Error: Please enter a question."
        
        llm = get_llm()
        
        retriever_obj = retriever(file)
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_obj,
            return_source_documents=True
        )
        
        response = qa.invoke({"query": query})
        return response['result']
        
    except Exception as e:
        return f"Error: {str(e)}"

rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count='single', file_types=['.pdf'], type='filepath'),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here....")
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG PDF Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)


rag_application.launch(server_name="0.0.0.0", server_port=7860)
