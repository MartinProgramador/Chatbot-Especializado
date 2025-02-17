import os
import sys
import argparse
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama
from pinecone import Pinecone, ServerlessSpec

pdf_path = "."

# Clases de excepción personalizadas
class ErrorProcesarFicheros(Exception):
    def __init__(self, parametro1):
        self.parametro1 = parametro1

class ExisteDirectorioSinFicheros(Exception):
    def __init__(self, parametro1):
        self.parametro1 = parametro1

class FileNotFoundErrorCustom(Exception):
    def __init__(self, parametro1):
        self.parametro1 = parametro1

# Función para cargar todos los PDFs de un directorio
def load_pdfs(pdf_path):
    try:
        if not os.path.isdir(pdf_path):
            raise FileNotFoundErrorCustom(f"El directorio {pdf_path} no se encuentra.")
    except FileNotFoundErrorCustom as e:
        print(str(e))
        sys.exit(1)

    documentos_pdf = []
    for filename in os.listdir(pdf_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(pdf_path, filename)
            try:
                reader = PdfReader(file_path)
                texto = " ".join([pagina.extract_text() for pagina in reader.pages])
                documentos_pdf.append(texto.replace("\n", " ").strip())
            except Exception as e:
                print(f"Error al procesar {filename}: {str(e)}")
    try:
        if not documentos_pdf:
            raise ExisteDirectorioSinFicheros("No hay ficheros que cargar.")
    except ExisteDirectorioSinFicheros as e:
        print(str(e))
        sys.exit(1)

    return "\n\n\n".join(documentos_pdf)

# Función para dividir el texto en chunks
def split_text_into_chunks(text, chunk_size=315, chunk_overlap=75):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    print("Troceando ficheros ...")
    return chunks

# Función para crear o cargar la base de datos en Pinecone
def create_or_load_pinecone_db(chunks, embeddings, index_name="local-rag-db"):
    # Obtén las variables de entorno para Pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") # tu api aqui 
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
    
    # Configuración para el servidor sin servidor de Pinecone
    spec = ServerlessSpec(cloud='aws', region='us-east-1')
    pc = Pinecone(api_key=PINECONE_API_KEY, spec=spec)

    # Verificar si el índice ya existe y, si no, crearlo
    existing_indexes = [index.name for index in pc.list_indexes()]
    if index_name not in existing_indexes:
        ejemplo_embedding = embeddings.embed_query("test")
        dim = len(ejemplo_embedding)
        print(f"Dimensión del embedding: {dim}")
        pc.create_index(name=index_name, dimension=dim, metric="cosine", spec=spec)
        print(f"Índice '{index_name}' creado.")
    else:
        print(f"Índice '{index_name}' ya existe.")

    index = pc.Index(index_name)

    # Crear documentos a partir de los chunks
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Generar embeddings para cada chunk
    vectors = embeddings.embed_documents([doc.page_content for doc in documents])
    ids = [str(i) for i in range(len(documents))]
    metadata = [{'text': doc.page_content} for doc in documents]

    # Upsert de los vectores en el índice
    index.upsert(vectors=[(ids[i], vectors[i], metadata[i]) for i in range(len(vectors))])
    print(f"Upserted {len(vectors)} vectores en el índice '{index_name}'.")
    return index, documents

# Función para recuperar documentos relevantes desde Pinecone
def get_relevant_documents(question, index, embeddings, top_k=10):
    query_vector = embeddings.embed_query(question)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    docs = [match['metadata']['text'] for match in results.get('matches', [])]
    if not docs:
        print("No se encontraron documentos relevantes.")
    else:
        print(f"Se encontraron {len(docs)} documentos relevantes.")
    return docs

# Función RAG para generar respuesta usando Ollama
def my_rag(contexts, question: str, model='llama3.2'):
    if not contexts:
        return "Lo siento, no tengo suficiente información para responder a esta pregunta."
    
    prompt_template = PromptTemplate(
        template="""
Utiliza la siguiente información para responder a la pregunta. Si la información no está en el contexto, responde "Lo siento, no tengo suficiente información para responder a esta pregunta.":

Contexto: {context}
---

Pregunta: {question}
Respuesta: Según el contexto,
""",
        input_variables=["context", "question"]
    )
    
    context = "\n\n---\n\n".join(contexts)
    formatted_prompt = prompt_template.format(context=context, question=question)
    
    print("Generando la respuesta...")
    response = ollama.generate(
        model=model,
        prompt=formatted_prompt,
        options={
            "temperature": 0.3,
            "top_k": 4,
            "num_predict": 1024
        }
    )
    
    return response['response']

# Función principal
def main():
    pdf_path = "."  # Directorio donde se encuentran los PDFs
    documentos_pdf = load_pdfs(pdf_path)
    print("Documentos pdf procesados correctamente.")

    chunks = split_text_into_chunks(documentos_pdf)
    
    # Crear el objeto de embeddings (se puede cambiar de modelo según convenga)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Crear o cargar la base de datos de Pinecone
    index, _ = create_or_load_pinecone_db(chunks, embeddings, index_name="local-rag-db")

    # Parsear la pregunta desde los argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Consulta el PDF procesado.")
    parser.add_argument("question", type=str, help="Pregunta a responder")
    args = parser.parse_args()
    question = args.question

    # Obtener documentos relevantes
    docs = get_relevant_documents(question, index, embeddings, top_k=10)
    
    # Generar la respuesta basada en los documentos encontrados
    answer = my_rag(docs, question)
    print(f"Respuesta: {answer}")

if __name__ == "__main__":
    main()
