import os
import sys
import argparse
import datetime
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

# Configuramos la API de Gemini
GOOGLE_API_KEY = "AIzaSyAWMmCjoPifpofExi5yv8rSphLidNCqxF4"

genai.configure(api_key=GOOGLE_API_KEY)

pdf_path = "pdf"

class error_procesar_ficheros(Exception):
    def __init__(self, parametro1):
        self.parametro1 = parametro1

class existe_directorio_sin_ficheros(Exception):
    def __init__(self, parametro1):
        self.parametro1 = parametro1

class file_not_found_error(Exception):
    def __init__(self, parametro1):
        self.parametro1 = parametro1
    
# Cargamos los pdf's
def load_pdfs(pdf_path):
    try:
        if not os.path.isdir(pdf_path):
                raise file_not_found_error(f"El directorio {pdf_path} no se encuentra.")
    except file_not_found_error as e:
                print(str(e))
                sys.exit(1)
    
    documentos_pdf = []
    for filename in os.listdir(pdf_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(pdf_path, filename)
                try:
                    reader = PdfReader(file_path)
                    todo_documento = " ".join([pagina.extract_text() for pagina in reader.pages])
                    documentos_pdf.append(todo_documento.replace("\n", " ").strip())
                except Exception as e:
                    print(f"Error al procesar {filename}: {str(e)}")
    try:
        if not documentos_pdf:
            raise existe_directorio_sin_ficheros(f"No hay ficheros que cargar")
    except existe_directorio_sin_ficheros as e:
            print(str(e))
            sys.exit(1)
    
    return "\n\n\n".join(documentos_pdf)

# Dividimos el texto de los pdf's en trozos
def split_text_into_chunks(documentos_pdf, chunk_size=1000, chunk_overlap=500):
    
    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""],chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(documentos_pdf)
    print("Troceando ficheros ...")
    return chunks

def configure_and_create_collection_chroma():
    
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

    # Buscamos los nombres de las colecciones en chroma
    #collection_names = chroma_client.list_collections()

    # Buscamos si ya existe la colección creada en otras ejecuciones previamente y la borramos
    #collection_name = chroma_client.get_collection(name="my_rag_db", embedding_function=embedding_function)
    #for collection_name in collection_names:
            #chroma_client.delete_collection(name=collection_name)

    # Generamos los embeddings
    # create_collection
    
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

    print(f"Configurando y creando la colección en ChromaDB ...")
    return collection

# Añadimos los datos de la colección a ChromaDB
def add_data_to_ChromaDB(chunks, collection):

    collection.add(
        documents=chunks,
        ids=[f"id{i}" for i in range(len(chunks))],
    )

    print("Datos añadidos a la colección")

    return collection

# Recuperamos los documentos de la colección relevantes de ChromaDB
def get_relevant_documents(question, collection):

    results = collection.query(
        query_texts=[question],
        n_results=10
    )

    print(f"Buscando los trozos más relevantes ...")

    return results['documents'][0]

# Generamos la respuesta en base al contexto, y a los documentos relevantes obtenidos
# de la BD
def my_rag(results, question: str):

    if not results:
        print(f"Lo siento, no tengo suficiente información para responder a esta pregunta.")

    prompt = PromptTemplate(
        template="""
        Si la información no está en el contexto, responde "Lo siento, no tengo suficiente información para responder a esta pregunta.":

        Contexto: {context}
        ---

        Pregunta: {question}
        Respuesta: Basa tu respuesta en "Según el contexto proporcionado," además de añadir la información más relevante.
        """,
        input_variables=["context", "question"]
    )

    context = "\n\n---\n\n".join(results)

    # Formateamos el prompt con el contexto y la pregunta
    formatted_prompt = prompt.format(context=context, question=question)

    # Generamos la respuesta usando el LLM
    genai.configure(api_key=GOOGLE_API_KEY)
    llm = genai.GenerativeModel(model_name='gemini-1.5-pro')

    response_text = llm.generate_content(
        formatted_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=1024,
            top_k=10,
            top_p=0.8
        )
    )

    print(f"Generando la respuesta ...")

    return response_text.text

def main():

    documentos_pdf = load_pdfs(pdf_path)
    print("Documentos pdf procesados correctamente")

    chunks = split_text_into_chunks(documentos_pdf)
    
    # Llamamos a Chroma para crear los embeddings y la colección
    collection = configure_and_create_collection_chroma()

    # Cargamos la collección en ChromaDB
    data_in_chroma = add_data_to_ChromaDB(chunks, collection)

    # Parseamos la pregunta
    parser = argparse.ArgumentParser(description="Consulta el PDF procesado.")
    parser.add_argument("question", type=str, help="Pregunta a responder")
    args = parser.parse_args()
    question = args.question

    if question:
        # Obtenemos los documentos más relevantes
        docs = get_relevant_documents(question, data_in_chroma)
        # Construímos la respuesta
        answer = my_rag(docs, question)
        # Mostramos la respuesta
        print(f"Respuesta: {answer}")

if __name__ == "__main__":
    main()




