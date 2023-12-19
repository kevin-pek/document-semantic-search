# Document Semantic Search

A simple Gradio interface for semantic search across multiple PDF documents using a combination of BM25 and vector embeddings to find relevant documents. The script builds a FAISS index on corpus of the uploaded documents, and first uses BM25 to find the top relevant results, then reranks them using cosine similarity to the search query.

## Setup

[Link to venv docs](https://docs.python.org/3/library/venv.html)

### Create environment

```shell
python3 -m venv venv
```

### To activate the environment

UNIX/MacOS:

```shell
source venv/bin/activate
```

Windows:

```shell
venv/Scripts/activate
```

### Install dependencies

If this is your first time running this or the package dependencies have changed, run this command to install all dependencies.

```shell
pip install -r requirements.txt
```

## Run

Run the app in reload mode with this command. This will let the app reload automatically when changes are made to the python script.

```shell
python main.py
```
