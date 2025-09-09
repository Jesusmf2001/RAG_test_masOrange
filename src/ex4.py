import os
from pathlib import Path
from dotenv import load_dotenv
import openai
from vector_db import VectorDDBB
from chatbot import Chatbot

if __name__ == "__main__":
    load_dotenv()
    client = openai.OpenAI(api_key=os.getenv("LITELLM_KEY"), base_url="https://llmproxy.ai.orange")

    # Cargar base de datos vectorial con la guía
    vector_db = VectorDDBB(client)
    vector_db.load_document_from_path(Path("../ai-engineer-evaluation-test.md"))

    # Crear instancia del chatbot
    chatbot = Chatbot(vector_db, client)

    pregunta = True 
    print("Si no desea hacer más preguntas responda \"salir\"")
    while pregunta:
        question = input("Pregunta: ")
        if question == "salir":
            pregunta = False
        answer = chatbot.ask_question(question)
        print("Respuesta:", answer)
        print("\n")
