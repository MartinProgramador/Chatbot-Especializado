import os
import sys
import argparse
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import easyocr


# Configurar la API de Gemini
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)

img_path = "."

class error_procesar_ficheros(Exception):
    def __init__(self, parametro1):
        self.parametro1 = parametro1

class existe_directorio_sin_ficheros(Exception):
    def __init__(self, parametro1):
        self.parametro1 = parametro1

class file_not_found_error(Exception):
    def __init__(self, parametro1):
        self.parametro1 = parametro1

# Extraer texto de PDFs
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
                todo_documento = " ".join([pagina.extract_text() for pagina in reader.pages if pagina.extract_text()])
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

# Extraer texto de imágenes usando EasyOCR
def extract_text_from_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"El archivo {image_path} no existe.")
    try:
        reader = easyocr.Reader(['es', 'en'])
        result = reader.readtext(image_path, detail=0, paragraph=True)
        texto = ' '.join(result)
        return texto.replace("\n", " ").strip()
    except Exception as e:
        raise ValueError(f"No se pudo extraer texto de la imagen: {e}")

# Dividir el texto en chunks
def split_text_into_chunks(documentos_pdf, chunk_size=1000, chunk_overlap=500):
    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(documentos_pdf)
    print("Troceando documentos ...")
    return chunks

# Configurar y crear la colección en ChromaDB
def configure_and_create_collection_chroma():
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
    except ValueError:
        pass

    collection = chroma_client.get_or_create_collection(
        name="local_rag_db", 
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine", "hnsw:search_ef": 300}
    )

    print(f"Configurando y creando la colección en ChromaDB...")
    return collection

# Añadir datos a ChromaDB
def add_data_to_ChromaDB(chunks, collection):
    collection.add(
        documents=chunks,
        ids=[f"id{i}" for i in range(len(chunks))],
    )

    print("Datos añadidos a la colección")
    return collection

# Recuperar documentos relevantes de ChromaDB
def get_relevant_documents(question, collection):
    results = collection.query(
        query_texts=[question],
        n_results=10
    )

    print(f"Buscando los trozos más relevantes...")
    return results['documents'][0]

# Generar respuesta usando Gemini
def my_rag(results, question: str):
    if not results:
        print(f"Lo siento, no tengo suficiente información para responder a esta pregunta.")
        return ""

    prompt_template = PromptTemplate(
        template="""
        Si la información no está en el contexto, responde "Lo siento, no tengo suficiente información para responder a esta pregunta.":

        Contexto: {context}
        ---

        Pregunta: {question}
        Respuesta: Según el contexto,
        """,
        input_variables=["context", "question"]
    )

    context = "\n\n---\n\n".join(results)
    formatted_prompt = prompt_template.format(context=context, question=question)

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

    print(f"Generando la respuesta...")
    return response_text.text

# Flujo principal
def main():
    parser = argparse.ArgumentParser(description="Consulta sobre documentos procesados (PDF o imagen).")
    parser.add_argument("file_path", type=str, help="Ruta al PDF o imagen")
    parser.add_argument("question", type=str, help="Pregunta a responder")
    args = parser.parse_args()

    # Extraer texto del archivo (PDF o imagen)
    if args.file_path.lower().endswith(".pdf"):
        print("Procesando archivo PDF...")
        documentos_pdf = load_pdfs(os.path.dirname(args.file_path))
    elif args.file_path.lower().endswith((".png", ".jpg", ".jpeg")):
        print("Procesando imagen...")
        documentos_pdf = extract_text_from_image(args.file_path)
    else:
        print("Formato de archivo no soportado. Usa PDF o imagen (PNG, JPG, JPEG).")
        sys.exit(1)

    if not documentos_pdf:
        print("No se pudo extraer texto del archivo.")
        return

    # Dividir el texto en chunks
    chunks = split_text_into_chunks(documentos_pdf)

    # Configurar y crear la colección en ChromaDB
    collection = configure_and_create_collection_chroma()

    # Cargar la colección en ChromaDB
    data_in_chroma = add_data_to_ChromaDB(chunks, collection)

    # Obtener documentos relevantes
    docs = get_relevant_documents(args.question, data_in_chroma)

    # Generar respuesta
    answer = my_rag(docs, args.question)

    # Mostrar la respuesta
    print(f"Respuesta: {answer}")

if __name__ == "__main__":
    main()
