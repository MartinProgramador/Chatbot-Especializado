# app.py
import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
import ollama
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import google.generativeai as genai
import easyocr

from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Configuraciones globales
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

global_documents = []  # Acumulamos los textos extraídos de los PDFs (y eventualmente imágenes)
pdf_filenames = []

# Configura tus API Keys para Google Generative AI y Pinecone
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "") # tu API aqui 
genai.configure(api_key=GOOGLE_API_KEY)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "") # tu API aqui 
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1-aws") 

### Funciones de utilidades

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text.replace("\n", " ").strip()
    except Exception as e:
        print(f"Error extrayendo PDF {file_path}: {e}")
        return ""

def extract_text_from_image(image_path):
    if not os.path.exists(image_path):
        return ""
    try:
        reader = easyocr.Reader(['es', 'en'])
        result = reader.readtext(image_path, detail=0, paragraph=True)
        text = ' '.join(result)
        return text.replace("\n", " ").strip()
    except Exception as e:
        print(f"Error extrayendo imagen {image_path}: {e}")
        return ""

def split_text_into_chunks(text, chunk_size=315, chunk_overlap=75):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def split_text_into_chunks_gemini(text, chunk_size=1000, chunk_overlap=500):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def split_text_into_chunks_ollama(text, chunk_size=900, chunk_overlap=350):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


### Pipelines

def pipeline_ollama_chroma(question, documents):
    if not documents:
        return "No hay documentos subidos."
    full_text = "\n\n\n".join(documents)
    chunks = split_text_into_chunks_ollama(full_text, chunk_size=900, chunk_overlap=350)
    # Configuramos la función de embeddings de Ollama
    embedding_function = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text",
    )
    
    try:
        chroma_client = chromadb.HttpClient(
            host="localhost",
            port=8000,
            settings=chromadb.config.Settings(
                persist_directory="./chroma_db",
            )
        )
    except ValueError:
        print("Error: No se puede conectar al servidor de Chroma. Asegúrate de que Docker esté en ejecución.")
        sys.exit(1)

    collection = chroma_client.create_collection(
        name="my_rag_db", 
        embedding_function=embedding_function,
        get_or_create=True,
        metadata= {
            "hnsw:space": "cosine", 
            "hnsw:search_ef": 300, 
            "description": "Colección ChromaDB usando Gemini"
        }
    )
    # Añadimos los chunks a la colección
    collection.upsert(
        documents=chunks,
        ids=[f"id{i}" for i in range(len(chunks))]
    )
    # Consultamos la colección
    results = collection.query(
        query_texts=[question],
        n_results=10,
    )
    relevant_docs = results['documents'][0]
    prompt_template = PromptTemplate(
        template="""Utiliza la siguiente información para responder a la pregunta. Si la información no está en el contexto, responde "Lo siento, no tengo suficiente información para responder a esta pregunta.":

        Contexto: {context}
        ---
        Pregunta: {question}
        Respuesta: Basa tu respuesta en "Según el contexto proporcionado," además de añadir la información más relevante.
        """,
        input_variables=["context", "question"]
    )
    context_str = "\n\n---\n\n".join(relevant_docs)
    formatted_prompt = prompt_template.format(context=context_str, question=question)
    response = ollama.generate(model='llama3.2', prompt=formatted_prompt,
                                options={"temperature": 0.3, "top_k": 10, "num_predict": 2048, "top_p": 0.9, "num_ctx": 2048,})
    return response.get('response', '')


def pipeline_ollama_pinecone(question, documents):
    if not documents:
        return "No hay documentos subidos."
    full_text = "\n\n\n".join(documents)
    chunks = split_text_into_chunks(full_text, chunk_size=315, chunk_overlap=75)
    # Usamos la nueva versión de HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Configuramos Pinecone sin usar pinecone.init
    spec = ServerlessSpec(cloud='aws', region='us-east-1')
    pc = Pinecone(api_key=PINECONE_API_KEY, spec=spec)
    index_name = "local-rag-db"
    existing_indexes = [index.name for index in pc.list_indexes()]
    if index_name not in existing_indexes:
        ejemplo_embedding = embeddings.embed_query("test")
        dim = len(ejemplo_embedding)
        pc.create_index(name=index_name, dimension=dim, metric="cosine", spec=spec)
    index = pc.Index(index_name)
    # Preparamos los documentos (usamos Document de langchain)
    docs_objs = [Document(page_content=chunk) for chunk in chunks]
    vectors = embeddings.embed_documents([doc.page_content for doc in docs_objs])
    ids = [str(i) for i in range(len(docs_objs))]
    metadata = [{'text': doc.page_content} for doc in docs_objs]
    index.upsert(vectors=[(ids[i], vectors[i], metadata[i]) for i in range(len(vectors))])
    # Consulta en Pinecone
    query_vector = embeddings.embed_query(question)
    results = index.query(vector=query_vector, top_k=10, include_metadata=True)
    matches = results.get('matches', [])
    docs = [match['metadata']['text'] for match in matches]
    prompt_template = PromptTemplate(
        template="""Utiliza la siguiente información para responder a la pregunta. Si la información no está en el contexto, responde "Lo siento, no tengo suficiente información para responder a esta pregunta.":

Contexto: {context}
---
Pregunta: {question}
Respuesta: Según el contexto,""",
        input_variables=["context", "question"]
    )
    context_str = "\n\n---\n\n".join(docs)
    formatted_prompt = prompt_template.format(context=context_str, question=question)
    response = ollama.generate(model='llama3.2', prompt=formatted_prompt,
                                options={"temperature": 0.3, "top_k": 4, "num_predict": 1024})
    return response.get('response', '')

def pipeline_gemini(question, documents):
    if not documents:
        return "No hay documentos subidos."
    full_text = "\n\n\n".join(documents)
    chunks = split_text_into_chunks_gemini(full_text, chunk_size=1000, chunk_overlap=500)
    
    embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=GOOGLE_API_KEY,
        model_name="models/embedding-001"
    )

    try:
        chroma_client = chromadb.HttpClient(
            host="localhost",
            port=8000,
            settings=chromadb.config.Settings(
                persist_directory="./chroma_db",
            )
        )
    except ValueError:
        print("Error: No se puede conectar al servidor de Chroma. Asegúrate de que Docker esté en ejecución.")
        sys.exit(1)
    
    collection = chroma_client.create_collection(
        name="my_rag_db", 
        embedding_function=embedding_function,
        get_or_create=True,
        metadata= {
            "hnsw:space": "cosine", 
            "hnsw:search_ef": 300, 
            "description": "Colección ChromaDB usando Gemini"
        }
    )

    collection.upsert(
        documents=chunks,
        ids=[f"id{i}" for i in range(len(chunks))]
    )
    results = collection.query(
        query_texts=[question],
        n_results=10
    )
    relevant_docs = results['documents'][0]
    prompt_template = PromptTemplate(
        template="""Si la información no está en el contexto, responde "Lo siento, no tengo suficiente información para responder a esta pregunta.":

        Contexto: {context}
        ---
        Pregunta: {question}
        Respuesta: Basa tu respuesta en "Según el contexto proporcionado," además de añadir la información más relevante. """,
                input_variables=["context", "question"]
    )
    context_str = "\n\n---\n\n".join(relevant_docs)
    formatted_prompt = prompt_template.format(context=context_str, question=question)
    genai.configure(api_key=GOOGLE_API_KEY)
    llm = genai.GenerativeModel(model_name='gemini-1.5-pro')
    response_text = llm.generate_content(
        formatted_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=1024,
            top_k=10,
            top_p=0.95
        )
    )
    return response_text.text


def pipeline_gemini_image(image_path, question):
    extracted_text = extract_text_from_image(image_path)
    if not extracted_text:
        return "No se pudo extraer texto de la imagen."
    chunks = split_text_into_chunks_gemini(extracted_text, chunk_size=1000, chunk_overlap=500)
    embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=GOOGLE_API_KEY,
        model_name="models/embedding-001"
    )
    chroma_client = chromadb.Client(
        chromadb.config.Settings(
            persist_directory="./path_to_chroma_db",
        )
    )
    try:
        chroma_client.delete_collection(name="local_rag_db")
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(
        name="local_rag_db", 
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine", "hnsw:search_ef": 300}
    )
    collection.upsert(
        documents=chunks,
        ids=[f"id{i}" for i in range(len(chunks))]
    )
    results = collection.query(
        query_texts=[question],
        n_results=10
    )
    relevant_docs = results['documents'][0]
    prompt_template = PromptTemplate(
        template="""Si la información no está en el contexto, responde "Lo siento, no tengo suficiente información para responder a esta pregunta.":

Contexto: {context}
---
Pregunta: {question}
Respuesta: Según el contexto,""",
        input_variables=["context", "question"]
    )
    context_str = "\n\n---\n\n".join(relevant_docs)
    formatted_prompt = prompt_template.format(context=context_str, question=question)
    genai.configure(api_key=GOOGLE_API_KEY)
    llm = genai.GenerativeModel(model_name='gemini-1.5-pro')
    response_text = llm.generate_content(
        formatted_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=1024,
            top_k=4,
        )
    )
    return response_text.text

### Endpoints de la API

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    engine = data.get("engine", "ollama_chroma")
    if not question:
        return jsonify({"answer": "Pregunta vacía."})
    
    if engine == "ollama_chroma":
        answer = pipeline_ollama_chroma(question, global_documents)
    elif engine == "ollama_pinecone":
        answer = pipeline_ollama_pinecone(question, global_documents)
    elif engine == "gemini":
        answer = pipeline_gemini(question, global_documents)
    else:
        answer = "Motor no reconocido."
    return jsonify({"answer": answer})

@app.route('/upload_pdfs', methods=['POST'])
def upload_pdfs():
    if 'files' not in request.files:
        return jsonify({"error": "No se recibieron archivos."}), 400
    files = request.files.getlist('files')
    uploaded_files = []
    for file in files:
        if file.filename == "":
            continue
        if file and file.filename.lower().endswith('.pdf'):
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            text = extract_text_from_pdf(file_path)
            if text:
                global_documents.append(text)
                uploaded_files.append(file.filename)
                pdf_filenames.append(file.filename)
    return jsonify({"files": uploaded_files})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No se recibió imagen."}), 400
    image_file = request.files['image']
    question = request.form.get("question", "")
    if image_file.filename == "":
        return jsonify({"error": "No se seleccionó imagen."}), 400
    filename = image_file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(file_path)
    answer = pipeline_gemini_image(file_path, question)
    extracted_text = extract_text_from_image(file_path)
    return jsonify({"answer": answer})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    global global_documents, pdf_filenames
    global_documents = []
    pdf_filenames = []
    for f in os.listdir(UPLOAD_FOLDER):
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        except Exception as e:
            print(f"Error borrando {f}: {e}")
    return jsonify({"status": "Historial borrado."})

@app.route('/')
def index():
    return send_from_directory('.', 'templates/chat.html')

if __name__ == '__main__':
    app.run(debug=True)
