from bs4 import BeautifulSoup
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from ollama import Client


# Cargar y procesar mensajes del archivo HTML
def load_messages_from_html(html_path):
    with open(html_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
    
    messages = []
    for message in soup.find_all("div", class_="message default clearfix"):
        # Extraer hora, remitente y contenido del mensaje
        time = message.find("div", class_="pull_right date details")
        sender = message.find("div", class_="from_name")
        text = message.find("div", class_="text")
        
        # Verificar que todos los elementos existen antes de usarlos
        if time and sender and text:
            time_text = time.get_text(strip=True)
            sender_text = sender.get_text(strip=True)
            message_text = text.get_text(strip=True)
            messages.append(f"[{time_text}] {sender_text}: {message_text}")
        else:
            print("Advertencia: Elemento faltante en un mensaje.")
    
    # Combinar los mensajes en un único texto
    return "\n".join(messages)


# Dividir el texto en chunks
def split_text_into_chunks(texto_completo, chunk_size=150, chunk_overlap=50):
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
    parser = argparse.ArgumentParser(description="Consulta los mensajes procesados.")
    parser.add_argument('pregunta', type=str, help="Pregunta para el LLM.")
    args = parser.parse_args()

    # Ruta del archivo HTML con mensajes de Telegram
    html_path = "" #Poner ruta correspondiente
    
    # Cargar los mensajes del archivo HTML
    texto_completo = load_messages_from_html(html_path)

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
    print(f"Respuesta: {answer}")


if __name__ == "__main__":
    main()
