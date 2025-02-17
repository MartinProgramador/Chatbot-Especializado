import os
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma  # Importación actualizada
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from PIL import Image
import easyocr  

# Configurar la API de Gemini
GOOGLE_API_KEY = "AIzaSyC1LJJ0WDktyDSA_NpPopOZbYRgvlCf4xM"  # Reemplaza con tu clave de API de Google

PROMPT_TEMPLATE = """
Basado en el contexto proporcionado, responde a la siguiente pregunta. Si la información no está en el contexto, di "No tengo suficiente información para responder a esta pregunta.":
    
Contexto: {context}
    
Pregunta: {question}
Respuesta:
"""

# Función para extraer texto de la imagen usando EasyOCR
def extract_text_from_image_easyocr(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"El archivo {image_path} no existe.")
    try:
        # Inicializar el lector de EasyOCR para español
        reader = easyocr.Reader(['es', 'en'])  # Puedes agregar más idiomas si es necesario
        result = reader.readtext(image_path, detail=0, paragraph=True)
        texto = ' '.join(result)
        return texto.replace("\n", " ").strip()
    except Exception as e:
        raise ValueError(f"No se pudo extraer texto de la imagen: {e}")

# Dividir el texto en chunks
def split_text_into_chunks(texto_completo, chunk_size=300, chunk_overlap=70):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(texto_completo)
    return chunks

# Procesar el texto y generar embeddings
def create_or_load_chroma_db(chunks, persist_directory="./path_to_chroma_db"):
    documents = [Document(page_content=chunk) for chunk in chunks]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    return vector_store

# Recuperar documentos relevantes
def get_relevant_documents(question, vector_db):
    retriever = vector_db.as_retriever()
    docs = retriever.invoke(question)
    return docs

def my_rag(docs, question: str):
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    prompt_template = PromptTemplate(
        template="""
        Si la información no está en el contexto, responde "No tengo suficiente información para responder a esta pregunta.":
    
        Contexto: {context}
        ---
    
        Pregunta: {question}
        Respuesta: Basado en el contexto proporcionado,
        """,
        input_variables=["context", "question"]
    )

    formatted_prompt = prompt_template.format(context=context, question=question)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY, temperature=0, max_tokens=1024)
    response_text = llm.invoke(formatted_prompt)
    return response_text.content

# Flujo principal
def main():
    parser = argparse.ArgumentParser(description="Consulta una imagen procesada.")
    parser.add_argument("image_path", type=str, help="Ruta a la imagen")
    parser.add_argument("question", type=str, help="Pregunta a responder")
    args = parser.parse_args()

    # Extraer texto de la imagen usando EasyOCR
    texto_completo = extract_text_from_image_easyocr(args.image_path)
    if not texto_completo:
        print("No se pudo extraer texto de la imagen.")
        return

    # Dividir el texto en chunks
    chunks = split_text_into_chunks(texto_completo)

    # Crear o cargar la base de datos de vectores
    persist_directory = "./path_to_chroma_db"
    vector_db = create_or_load_chroma_db(chunks, persist_directory=persist_directory)

    # Obtener documentos relevantes
    docs = get_relevant_documents(args.question, vector_db)

    if not docs:
        print("No se encontraron documentos relevantes.")
        return

    # Generar la respuesta
    answer = my_rag(docs, args.question)

    # Mostrar la respuesta
    print(f"Respuesta: {answer}")
    sources = os.path.basename(args.image_path)
    print(f"Origen de datos: {sources}")

if __name__ == "__main__":
    main()
