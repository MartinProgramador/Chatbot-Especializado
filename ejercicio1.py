import argparse
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from ollama import Client
import pinecone


# Cargar el contenido del PDF
def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    texto_completo = ""
    for pagina in reader.pages:
        texto_completo += pagina.extract_text()
    
    # Limpiar el texto: eliminar caracteres innecesarios
    texto_completo = texto_completo.replace('\n', ' ')  # Reemplazar saltos de línea por espacio
    texto_completo = texto_completo.replace("17_10_2024_Introducción a SAP SD.pdf:2:1", "")  # Depende de cada caso (esto para el enunciado de la práctica)
    return texto_completo


# Dividir el texto en chunks
def split_text_into_chunks(texto_completo, chunk_size=200, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(texto_completo)
    print(f"Número de chunks creados: {len(chunks)}")
    return chunks


# Crear o cargar la base de datos Chroma
def create_or_load_chroma_db(chunks, persist_directory="./chroma_db"):
    # Crear documentos a partir de los chunks
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Crear embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

    # Crear y persistir la base de datos Chroma si no existe
    chroma_db = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory, collection_name='local_rag_db')
    return chroma_db


# Recuperar documentos relevantes de la base de datos Chroma
def get_relevant_documents(question, vector_db):
    retriever = vector_db.as_retriever()
    docs = retriever.invoke(question)
    # print(f"Documents fetched from database: {len(docs)}")
    print(f"Número de documentos relevantes recuperados: {len(docs)}")
    return docs


# Generar respuesta usando Ollama
def generate_answer(docs, question, model='llama3'):
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Crear un prompt RAG
    formatted_prompt = f"""Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """
    
    # Llamar a la API de Ollama para generar la respuesta
    ollama_host_url = "http://localhost:11434"
    client = Client(host=ollama_host_url)
    response = client.chat(model=model, messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']


# Función principal
def main():
    parser = argparse.ArgumentParser(description="Consulta el PDF procesado.")
    parser.add_argument('pregunta', type=str, help="Pregunta para el LLM.")
    args = parser.parse_args()

    # Path del PDF
    pdf_path = "Enunciado_TIC_SAP.pdf"
    
    # Cargar el contenido del PDF
    texto_completo = load_pdf(pdf_path)

    # Dividir el texto en chunks
    chunks = split_text_into_chunks(texto_completo)

    # Crear o cargar la base de datos de Chroma
    chroma_db = create_or_load_chroma_db(chunks)

    # Cargar la base de datos Chroma
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434", show_progress=True), collection_name="local_rag_db")

    # Obtener documentos relevantes
    docs = get_relevant_documents(args.pregunta, vector_db)

    # Generar la respuesta
    answer = generate_answer(docs, args.pregunta)

    # Mostrar la respuesta
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()