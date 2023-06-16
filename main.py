from haystack.nodes import PreProcessor, PDFToTextConverter, EmbeddingRetriever, TransformersReader
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import DocumentSearchPipeline, ExtractiveQAPipeline
import gradio as gr

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="word",
    split_length=100,
    split_respect_sentence_boundary=True,
    split_overlap=3
)
document_store = InMemoryDocumentStore(embedding_dim=384)
reader = TransformersReader("sentence-transformers/all-MiniLM-L6-v2")
retriever = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L6-v2")
pipeline = ExtractiveQAPipeline(reader, retriever)
converter = PDFToTextConverter(remove_numeric_tables=True)

def print_answers(results):
    fields = ["answer", "score"] # "context"
    answers = results["answers"]
    filtered_answers = []
    for ans in answers:
        filtered_ans = {
            field: getattr(ans, field) for field in fields if getattr(ans, field) is not None
        }
        filtered_answers.append(filtered_ans)
    return filtered_answers

def write_pdf(pdf_file):
    document = converter.convert(file_path=pdf_file.name, meta=None)[0]
    preprocessed_docs = preprocessor.process(document)
    document_store.write_documents(preprocessed_docs)
    document_store.update_embeddings(retriever)

def predict(question, pdf_file):
    write_pdf(pdf_file)
    result = pipeline.run(query=question, params={"Retriever": { "top_k": 2 }})
    answers = print_answers(result)
    return answers

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
