# 💼 Nombre del proyecto

<h3 align="center" style=" line-height: 1.375rem; letter-spacing: 0.0091428571em; font-style: italic;">RAG AGAINST THE MACHINE</h3>

## 📝 Descripción

Desarrollar e implementar un sistema que facilite a los clientes finales realizar preguntas en lenguaje humano natural respecto a asuntos vinculados con la empresa (y lograr una respuesta razonable). 

## 📊 Guía de configuración y ejecución del sistema

  ### 💻 Prerrequisitos
-	Clonar el repositorio.
-	Tener Visual Studio Code instalado.
-	Tener una versión de Python 3.12.5 o superior.
-	Instalar Ollama.
-	Instalar Docker.
-	Descargar Telegram Desktop
-	Será necesario tener pip instalado en el sistema para gestionar la instalación de las dependencias requeridas por el proyecto.
-	Es recomendable ejecutar la práctica en un entorno virtual **(venv)**, para eso:
    - Abrir una terminal de *Windows, Mac o Linux*.
    - Ir hasta el directorio del proyecto, usando: *cd /ruta/al/proyecto*
    - Crear el entorno virtual: python -m venv nombre
    - Activar el entorno:
      - En Windows: *nombre\Scripts\activate*
      - En macOS/Linux: *source nombre/bin/activate*
    - Comprobar que el entorno virtual está activado. Se mostrará el nombre del entorno al comienzo de la línea de comandos entre paréntesis.
    - Instalar las dependencias necesarias para ejecutar cualquier ejercicio.
    - Ya se podrá ejecutar cualquier ejercicio como se indica más abajo, usando *python3 nombre_fichero.py*
    - Una vez finalizado, desactivar el entorno virtual, usando el comando *deactivate*

## 💻 Pasos para la ejecución del sistema

> [!NOTE]
> Se podrá ejecutar cada ejercicio de manera individual como se explica en cada uno de los apartados siguientes, a excepción del *Ejercicio 1 opcional* o se podrán ejecutar a través de una *interfaz* usando el framework de *Flask*, para poder llevarlo a cabo es necesario, lo siguiente:
> 1. Abrir un terminal y situarse en la ruta del proyecto.
> 2. Instalar Flask, utilizando: *pip3 install Flask*.
> 3. Instalar las dependencias necesarias, sino se han instalado previamente, al ejecutar cualquier ejercicio de manera individual.
> 4. A continuación debemos ejecutar la aplicación *Flask*, poniendo: *python3 -m flask run --debug*, esto nos permitirá que cualquier cambio realizado en el código, en el fichero *chat.html*, o en *(estilos.css)* o en *app.py* se refresque automáticamente.
> 5. Abrir un navegador y introducir la ruta: *http://127.0.0.1:5000*.
> 6. Para probar la aplicación es necesario, elegir el modelo de entre 3 posibles opciones: *Ollama-Chroma, Ollama-Pinecone y Gemini*. La funcionalidad de cargar una imagen funcionará con el modelo *Gemini* y lo primero será escribir la pregunta y luego subir la imagen, de esta manera el modelo nos responderá a dicha petición. Por otro lado, tenemos la posibilidad de subir documentos en formato **PDF** y hacer preguntas sobre los mismos, seleccionando el modelo de procesamiento deseado.

Una vez que se ha clonado el repositorio será necesario configurar las dependencias para que cada ejercicio se pueda ejecutar correctamente. Seguidamente y por cada ejercicio desarrollado, se mencionan como instalarse.

### 💻 Ejercicio 1: Parte Obligatoria
1.	Instalar las siguientes dependencias:

-	pip install PyPDF2 
-	pip install langchain
-	pip install langchain-core
-	pip install ollama
-	pip install chromadb
  
Modificar la ruta para cargar los documentos: **variable (pdf_path)** del fichero *ejercicio1.py*

2.	Para reproducir los experimentos realizados tenemos que:
-	Modificar los parámetros “chunk_size” y “chunk_overlap”, para modificar el tamaño de cada trozo y su solapamiento, en la función “split_text_into_chunks”
-	Modificar tanto el número de resultados relevantes (n_results) así como al generar la respuesta (temperature, top_k, num_predict, top_p y num_ctx) para ver cómo poder simular todo lo que se ha probado para conseguir un resultado óptimo y que así se ha reflejado en la memoria.

Para ejecutarlo por terminal sería suficiente, con poner: *python3 ejercicio1.py “pregunta”*

> [!NOTE]
> En este caso, será necesario:
> - Abrir un terminal y ejecutar: *ollama serve*
> - Abrir un terminal de Docker y ejecutar: *docker run -p 8000:8000 chromadb/chroma*, para iniciar el servidor de **ChromaDB**

### 💻 Ejercicio 2: Parte Obligatoria
1.	Instalar las siguientes dependencias:

Para poder ejecutar este ejercicio necesitamos disponer de la *API_KEY* de *Google*, para ello tenemos que ir al enlace: https://ai.google.dev/gemini-api/docs/api-key?hl=es-419 y configurarla. A continuación instalar la dependencias, como se indica a continuación:

-	pip install google-generativeai

Modificar la ruta para cargar los documentos: **variable (pdf_path)** del fichero *ejercicio2.py*

2.	Para reproducir los experimentos realizados tenemos que:

-	Modificar los parámetros “chunk_size” y “chunk_overlap”, para modificar el tamaño de cada trozo y su solapamiento, en la función “split_text_into_chunks”
-	Modificar tanto el número de resultados relevantes (n_results) así como al generar la respuesta (temperature, top_k, max_output_tokens y top_p) para ver cómo poder simular todo lo que se ha probado para conseguir un resultado óptimo y que así se ha reflejado en la memoria.

Para ejecutarlo por terminal sería suficiente, con poner: *python3 ejercicio2.py “pregunta”*

> [!NOTE]
> En este caso, se necesitará:
> - Abrir un terminal de Docker y ejecutar: *docker run -p 8000:8000 chromadb/chroma*

### 💻 Ejercicio 1: Parte Opcional
1.	Instalar las siguientes dependencias:
-	pip install beautifulsoup4
-	pip install langchain-chroma
-	pip install langchain-community

2.	Modificar la ruta **variable (htlm_path)** del archivo *chatbot_telegram.py* para que los mensajes del archivo HTML se carguen correctamente.

3. Ejecutar el servidor de embeddings y ChromaDB
- Iniciar Ollama en una terminal:
  ollama serve

4. Ejecutar el chatbot con una consulta
   python chatbot_telegram.py *“pregunta”*

Para reproducir los experimentos y optimizar los resultados, puede modificar los siguientes parámetros:

1️⃣ Ajustar el tamaño y solapamiento de los fragmentos (chunks)

Ubicado en split_text_into_chunks() en chatbot_telegram.py:
def split_text_into_chunks(texto_completo, chunk_size=150, chunk_overlap=50):
- chunk_size: Controla el tamaño de cada fragmento de texto almacenado en ChromaDB.
- chunk_overlap: Define el solapamiento entre fragmentos para mejorar la recuperación de información.
Pruebe diferentes valores y analice cómo afecta la precisión de las respuestas.

2️⃣ Ajustar la cantidad de documentos recuperados de ChromaDB:

En get_relevant_documents(), puedes cambiar el número de fragmentos relevantes que se recuperan al hacer una consulta:
retriever = vector_db.as_retriever(search_kwargs={"k": 5})
- k: Número de fragmentos que se recuperan para generar la respuesta.
Reducir k puede acelerar la respuesta, pero puede perder información relevante.

3️⃣ Ajustar la generación de respuestas con LLaMA 3:

En generate_answer(), se pueden modificar los parámetros del prompt enviado al modelo:

formatted_prompt = f"""Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

Cambiar la estructura del prompt puede mejorar la coherencia de las respuestas.

🛠 Posibles Errores y Soluciones:

El chatbot no encuentra información relevante	-> Ajustar chunk_size y chunk_overlap para mejorar la segmentación del texto.

Error al procesar el archivo HTML ->	Verificar la estructura del HTML y que los elementos (from_name, text, date) estén correctamente extraídos.

ChromaDB no devuelve fragmentos precisos ->	Ajustar k en la función de recuperación de documentos.

LLaMA 3 responde con información irrelevante -> Modificar el formatted_prompt para que dependa más del contexto.



### 💻 Ejercicio 2: Parte Opcional

1.	Instalar las siguientes dependencias:

-	pip install Transformers
-	pip install pinecone-client

Modificar la ruta para cargar los documentos: **variable (pdf_path)** del fichero *ejercicio_pinecone.py*

2.	Para poder ejecutar este ejercicio necesitamos disponer de la *API_KEY* de *Pinecone*, para ello tenemos que ir al enlace: https://docs.pinecone.io/guides/get-started/quickstart  y configurarla en el apartado “Get an-API Key”. Con esto ya tendremos funcionando a *Pinecone* como BD vectorial en la nube.

3.	Para reproducir los experimentos realizados debemos de:

-	Modificar los parámetros “chunk_size” y “chunk_overlap”, para modificar el tamaño de cada trozo y su solapamiento, en la función “split_text_into_chunks”
-	Modificar tanto el número de resultados relevantes (n_results) así como al generar la respuesta (temperatura, top_k,num_predict y top_p) y ver cómo poder simular todo lo que se ha probado para conseguir un resultado óptimo y que así se ha reflejado en la memoria.

Para ejecutarlo por terminal sería suficiente, con poner: *python3 ejercicio_pinecone.py “pregunta”*

### 💻 Ejercicio 3: Parte Opcional
1.	Instalar las siguientes dependencias:

Para poder ejecutar este ejercicio necesitamos disponer de la *API_KEY* de *Google*, para ello tenemos que ir al enlace: https://ai.google.dev/gemini-api/docs/api-key?hl=es-419 y configurarla. A continuación instalar la dependencias, como se indica a continuación:

-	pip install google-generativeai
-	pip install argparse
-	pip install easyocr

Proporcionar una ruta válida para la **variable (img_path)**, en el archivo *ejercicio_imagen* que haga referencia a un archivo que contenga alguna imagen en el sistema del usuario, es decir que ese archivo exista en la ubicación especificada. Además la imagen debe estar en un formato que *EasyOcr* pueda leerlo (como .jpg, .png, etc…)

> [!NOTE]
> El usuario necesitará tener permisos de lectura para el/los archivo/s de imagen.

2.	Para reproducir los experimentos realizados debemos de poner: *python3 "nombre_imagen" ejercicio_imagen.py “pregunta”*

