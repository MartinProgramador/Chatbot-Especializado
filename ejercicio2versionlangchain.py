from langchain import hub
import os
import argparse
from PyPDF2 import PdfReader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

# Configurar la API de Gemini
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)

# Ruta al PDF
pdf_path = "pdf/archivo.pdf"

# Función para cargar el PDF
def load_pdf(pdf_path):
    if os.path.exists(pdf_path):
        reader = PdfReader(pdf_path)
        texto_completo = ""
        for pagina in reader.pages:
            texto_completo += pagina.extract_text()
    
        texto_completo = texto_completo.replace('\n', ' ') 
        texto_completo = texto_completo.replace("archivo:2:1", "")
        return texto_completo
    else:
        raise FileNotFoundError(f"El archivo {pdf_path} no existe.")

# Dividir el texto en chunks
def split_text_into_chunks(texto_completo, chunk_size=100, chunk_overlap=70):
    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""],chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(texto_completo)
    print(f"Número de chunks creados: {len(chunks)}")
    return chunks

# Procesar el PDF y generar embeddings
def create_or_load_chroma_db(chunks, persist_directory="./chroma_db"):
    documents = [Document(page_content=chunk) for chunk in chunks]

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    chroma_db = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings,  
        persist_directory=persist_directory,
        collection_name='local_rag_db'
    )

    return chroma_db

# Recuperar documentos relevantes de la base de datos Chroma
def get_relevant_documents(question, vector_db):
    retriever = vector_db.as_retriever()
    docs = retriever.invoke(question)
    return docs

def my_rag(docs, question: str, persist_directory="./chroma_db"):
    
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    formatted_prompt = f"""Responde en base al contexto:
    {context}
    Question: {question}
    """

    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

    results = vector_db.similarity_search(question, k=10)

    if not results:
        print("No se encontraron documentos relevantes.")
        return

    formatted_prompt = formatted_prompt.format(context=context, question=question)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    
    response_text = llm.invoke(formatted_prompt)

    vector_db.get()['documents']

    return response_text.content

def main():
    # Cargar el contenido del PDF
    texto_completo = load_pdf(pdf_path)

    # Dividir el texto en chunks
    chunks = split_text_into_chunks(texto_completo)

    # Parsear la pregunta
    parser = argparse.ArgumentParser(description="Consultar el PDF procesado.")
    parser.add_argument("pregunta", type=str, help="Pregunta para el LLM")
    args = parser.parse_args()
    pregunta = args.pregunta

    # Crear o cargar la base de datos de Chroma
    chroma_db = create_or_load_chroma_db(chunks)

    vector_db = Chroma(client_settings=chroma_db, persist_directory="./chroma_db", embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY))

    if pregunta:
        # Obtener documentos relevantes
        docs = get_relevant_documents(args.pregunta, vector_db)
        answer = my_rag(docs, pregunta)
        # Mostrar la respuesta
        print(f"Respuesta: {answer}")

if __name__ == "__main__":
    main()