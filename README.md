# Price Online Comparator (POC)

Esta aplicación Streamlit digitaliza el Price Online, permitiendo cargar ficheros Excel con fichas de equipamiento de vehículos, compararlos y consultar diferencias asistido por IA.

## ¿Qué hace `app.py`?

El script principal implementa toda la lógica de la herramienta:

- **Carga y normaliza datos**: lee hojas de cálculo (`pandas`) sin depender de un formato rígido, detecta cabeceras, limpia textos y agrupa filas en secciones canónicas.
- **Estructura la información**: genera un CSV normalizado con metadatos (mercado, variante, marca, valores numéricos, unidades estimadas, marcas de referencia, timestamp de ingestión, etc.).
- **Compara variantes**: permite elegir dos versiones (del mismo fichero o de ficheros diferentes), colapsa valores duplicados y calcula diferencias sección por sección.
- **Genera informes**: construye un informe en Markdown y tablas con diferencias entre variantes, y prepara datos exportables (`build_export_filename`).
- **Chat asistido por IA**: usa un flujo RAG ligero que selecciona filas relevantes y las pasa a OpenAI/crewai para responder preguntas de negocio con streaming en la interfaz.
- **Interfaz Streamlit**: define un layout de dos columnas (panel principal + chat), controles para subir ficheros, seleccionar hojas, variantes, secciones y mostrar resultados.

## Requisitos previos

- Python 3.10 o superior.
- Una clave válida de la API de OpenAI.

## Instalación

1. Clona este repositorio y accede a la carpeta del proyecto.
2. Crea y activa un entorno virtual (opcional pero recomendado):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Crea un fichero `.env` en la raíz con tu clave de OpenAI:

   ```ini
   OPENAI_API_KEY="tu_clave"
   ```

   > El script abortará con un mensaje de error si la variable no está disponible.

## Ejecución

Lanza la aplicación con Streamlit:

```bash
streamlit run app.py
```

La interfaz permite dos modos de comparación:

- **Same file**: subir un único Excel y comparar dos variantes dentro de una misma hoja.
- **Cross file**: subir dos Excels distintos y confrontar variantes entre ellos.

El panel derecho muestra un historial de chat con respuestas generadas en streaming por la IA (vía OpenAI + crewAI) usando como contexto las filas relevantes del CSV.

## Estructura de archivos

- `app.py`: lógica completa de ingestión, comparación, informes y UI.
- `requirements.txt`: dependencias Python necesarias para ejecutar la aplicación.
- `.streamlit/` (opcional): si se desea añadir configuraciones adicionales de Streamlit.

## Notas adicionales

- Los ficheros Excel originales no se almacenan; sólo se generan tablas en memoria y CSV descargables desde la interfaz.
- El parseo intenta ser tolerante con formatos heterogéneos, pero conviene que las hojas mantengan la columna `Feature/Attribute` y valores consistentes.
- El modelo por defecto es `gpt-4o`, puedes modificarlo en la llamada `generate_agent_answer` si tu plan de OpenAI difiere.

## Licencia

Pendiente de definir.
