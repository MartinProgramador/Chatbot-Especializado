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


## 💻 Pasos para la ejecución del sistema

> [!NOTE]
> Tenemos que situarnos en la ruta correspondiente a cada ejercicio para poder ejecutarlo, sino se producirá un error.
Una vez instalado es necesario configurar las dependencias para que cada ejercicio se pueda ejecutar correctamente. Seguidamente y por cada ejercicio desarrollado, se mencionan como instalarse.

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
-	Modificar tanto el número de resultados relevantes (n_results) así como al generar la respuesta (temperatura, top_k,num_predict y top_p) y ver cómo poder simular todo lo que se ha probado para conseguir un resultado óptimo y que así se ha reflejado en la memoria.

Para ejecutarlo por terminal sería suficiente, con poner: *python3 ejercicio1.py “pregunta”*

> [!NOTE]
> En este caso, será necesario:
> - Abrir un terminal y ejecutar: *ollama serve*
> - Abrir un terminal de Docker y ejecutar: *docker run -p 8000:8000 chromadb/chroma*, para iniciar el servidor de **ChromaDB**

### 💻 Ejercicio 2: Parte Obligatoria
1.	Instalar las siguientes dependencias:

Para poder ejecutar este ejercicio necesitamos disponer de la API_KEY de Google, para ello tenemos que ir al enlace: https://ai.google.dev/gemini-api/docs/api-key?hl=es-419 y configurarla. A continuación instalar la dependencias, como se indica a continuación:

-	pip install google-generativeai

Modificar la ruta para cargar los documentos: **variable (pdf_path)** del fichero *ejercicio2.py*

2.	Para reproducir los experimentos realizados tenemos que:

-	Modificar los parámetros “chunk_size” y “chunk_overlap”, para modificar el tamaño de cada trozo y su solapamiento, en la función “split_text_into_chunks”
-	Modificar tanto el número de resultados relevantes (n_results) así como al generar la respuesta (temperatura, top_k,num_predict y top_p) y ver cómo poder simular todo lo que se ha probado para conseguir un resultado óptimo y que así se ha reflejado en la memoria.

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

4.	Para reproducir los experimentos realizados debemos de:


### 💻 Ejercicio 2: Parte Opcional

1.	Instalar las siguientes dependencias:

-	pip install Transformers
-	pip install pinecone-client

Modificar la ruta para cargar los documentos: **variable (pdf_path)** del fichero *main_pinecone.py*

2.	Para poder ejecutar este ejercicio necesitamos disponer de la *API_KEY* de *Pinecone*, para ello tenemos que ir al enlace: https://docs.pinecone.io/guides/get-started/quickstart  y configurarla en el apartado “Get an-API Key”. Con esto ya tendremos funcionando a *Pinecone* como BD vectorial en la nube.

3.	Para reproducir los experimentos realizados debemos de:

-	Modificar los parámetros “chunk_size” y “chunk_overlap”, para modificar el tamaño de cada trozo y su solapamiento, en la función “split_text_into_chunks”
-	Modificar tanto el número de resultados relevantes (n_results) así como al generar la respuesta (temperatura, top_k,num_predict y top_p) y ver cómo poder simular todo lo que se ha probado para conseguir un resultado óptimo y que así se ha reflejado en la memoria.

Para ejecutarlo por terminal sería suficiente, con poner: *python3 ejercicio2_pinecone.py “pregunta”*

### 💻 Ejercicio 3: Parte Opcional
1.	Instalar las siguientes dependencias:

-	pip install langchain-google-genai
-	pip install Pillow
-	pip install easyocr

Proporcionar una ruta válida para la **variable (image_path)**, en el archivo *ejercicio2_versionimg* que haga referencia a un archivo de imagen en el sistema del usuario, es decir que ese archivo exista en la ubicación especificada. Además la imagen debe estar en un formato que *EasyOcr* pueda leerlo (como .jpg, .png, etc…)

> [!NOTE]
> El usuario necesitará tener permisos de lectura para el/los archivo/s de imagen.

2.	Para reproducir los experimentos realizados debemos de:
