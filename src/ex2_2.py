
import os
from dotenv import load_dotenv
import openai
from vector_db import VectorDDBB

if __name__ == "__main__":
    # Cargar variables de entorno
    load_dotenv()
    api_key = os.getenv("LITELLM_KEY")

    # Crear cliente OpenAI
    client = openai.OpenAI(api_key=api_key, base_url="https://llmproxy.ai.orange")

    # Crear instancia de la base de datos vectorial
    vector_db = VectorDDBB(client)

    # Markdown de ejemplo
    sample_markdown = """
    # Introduction
    This is the introduction section.
    
    ## First Section
    This is the first section with some content.
    
    ## Second Section
    This is the second section with different content.
    
    ## Third Section
    And this is the third section.
    """
    
    # Cargar documento y crear embeddings
    vector_db.load_document(sample_markdown)
    
    # Imprimir número de embeddings
    vector_db.print_number_of_embeddings()
    # Output esperado: 4
