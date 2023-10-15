import os
import sys

from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import gradio as gr

# Enable to cache & reuse the model to disk (for repeated queries on the same data)
PERSIST = False

query = sys.argv[1]
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  raw_documents = DirectoryLoader("persist").load()
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  documents = text_splitter.split_documents(raw_documents)
  vectorstore = FAISS.from_documents(documents, embedding=embeddings)
  from langchain.indexes.vectorstore import VectorStoreIndexWrapper
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  loader = TextLoader('data.txt')
  # This code can also import folders, including various filetypes like PDFs using the DirectoryLoader.
  # loader = DirectoryLoader(".", glob="*.txt")
  if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
  else:
    index = VectorstoreIndexCreator().from_loaders([loader])

chain = RetrievalQA.from_chain_type(
  llm=,
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 5}),
)
print(chain.run(query))

interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.components.Textbox(lines = 1, label="Enter your search query here..."),
        gr.components.File(file_count="single", type="file", label="Upload a file here.")
    ],
    outputs="text",
    title="Search",
    interpretation=None,
    theme="default" # “default", “huggingface", “dark-grass", “peach"
)

interface.launch()
