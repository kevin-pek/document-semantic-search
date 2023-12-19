from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.retrievers import BM25Retriever
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.vectorstores import FAISS
import gradio as gr
import re

model = "msmarco-distilbert-base-v4"
embeddings = SentenceTransformerEmbeddings(model_name=model)
prev_files = None
retriever = None

def handle_files_and_query(query, files):
    results = ""
    global prev_files, retriever
    if files is not None and files != prev_files:
        documents = []
        prev_files = files
        for file in files:
            documents.extend(PyMuPDFLoader(file).load_and_split(SentenceTransformersTokenTextSplitter(model_name=model)))
        retriever = BM25Retriever.from_documents(documents, k=10)
        results += "Index created successfully!\n"
        print("Index created successfully!")
    elif files is None:
        print("No files uploaded.")
    else:
        print("Reusing index since no files changed.")

    print(f"Query: {query}")
    if query:
        search_results = retriever.get_relevant_documents(query, k=25)
        pattern = r'[^\\/]+$' # pattern to get filename from filepath
        reranked_results = FAISS.from_documents(search_results, embeddings, distance_strategy=DistanceStrategy.COSINE).similarity_search(query, k=1)
        print([
            f"Source: {re.search(pattern, result.metadata['file_path']).group(0)}\nPage: {result.metadata['page']}"
            for result in reranked_results
        ][0])
        results = [
            f"Source: {re.search(pattern, result.metadata['file_path']).group(0)}\nPage: {result.metadata['page']}\nContent:\n{result.page_content}"
            for result in reranked_results
        ][0]
    return results

interface = gr.Interface(
    fn=handle_files_and_query,
    inputs=[
        gr.Textbox(lines = 1, label="Enter your search query here..."),
        gr.File(file_count="multiple", type="filepath", file_types=[".pdf"], label="Upload a file here.")
    ],
    outputs="text",
    title="Similarity Search for PDFs"
)

interface.launch()
