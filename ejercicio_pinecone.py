import os
import argparse
import sys
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
import ollama
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

pdf_path = "."


class error_procesar_ficheros(Exception):
    def __init__(self, parametro1):
        self.parametro1 = parametro1

class existe_directorio_sin_ficheros(Exception):
    def __init__(self, parametro1):
        self.parametro1 = parametro1

class file_not_found_error(Exception):
    def __init__(self, parametro1):
        self.parametro1 = parametro1

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

# Dividir el texto en chunks

def split_text_into_chunks(documentos_pdf, chunk_size=315, chunk_overlap=75):
    
    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""],chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(documentos_pdf)
    print("Troceando ficheros ...")
    return chunks

def configure_and_create_collection_chroma():
    
    embedding_function = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text",
    )

    chroma_client = chromadb.Client(
        chromadb.config.Settings(
            persist_directory="./path_to_chroma_db",
        )
    )

    # Eliminamos la colección si ya existe
    try:
        chroma_client.delete_collection(name="local_rag_db")
    except ValueError:
        # La colección no existía previamente
        pass

    collection = chroma_client.get_or_create_collection(
        name="local_rag_db", 
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine", "hnsw:search_ef": 300}
    )

    print(f"Configurando y creando la colección en ChromaDB...")
    return collection

def add_data_to_ChromaDB(chunks, collection):

    collection.add(
        documents=chunks,
        ids=[f"id{i}" for i in range(len(chunks))],
    )

    print("Datos añadidos a la colección")

    return collection

# Recuperar documentos relevantes de la base de datos ChromaDB
def get_relevant_documents(question, collection):

    results = collection.query(
        query_texts=[question],
        n_results=10,
    )
    
    print(f"Buscando los trozos más relevantes...")

    return results['documents'][0]

def my_rag(results, question: str, model='llama3.2'):

    if not results:
        print(f"Lo siento, no tengo suficiente información para responder a esta pregunta.")
    
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

    context = "\n\n---\n\n".join(results)

    formatted_prompt = prompt_template.format(context=context, question=question)
    
    # Generar la respuesta usando el formatted_prompt
    response = ollama.generate(model=model, prompt=formatted_prompt,
        options={
            "temperature": 0.3,
            "top_k": 4,
            'num_predict': 1024
        }
    )

    print(f"Generando la respuesta...")
    
    return response['response']

def main():

    documentos_pdf = load_pdfs(pdf_path)
    print("Documentos pdf procesados correctamente")

    chunks = split_text_into_chunks(documentos_pdf)

    # Llamar a chroma
    collection = configure_and_create_collection_chroma()

    # Cargar la colección
    data_in_chroma = add_data_to_ChromaDB(chunks, collection)

    # Parsear la pregunta
    parser = argparse.ArgumentParser(description="Consulta el PDF procesado.")
    parser.add_argument("question", type=str, help="Pregunta a responder")
    args = parser.parse_args()
    question = args.question

    if question:

        # Obtener documentos relevantes
        #docs = get_relevant_documents(args.question, data_in_chroma)
        # sources
        docs = get_relevant_documents(question, data_in_chroma)
        answer = my_rag(docs, question)
        print(f"Respuesta: {answer}")

# Función principal
if __name__ == "__main__":
    main()
