import os
import openai
from dotenv import load_dotenv

# Cargamos la api key desde nuestro .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Se crea cliente OpenAI
client = openai.OpenAI(
    api_key=api_key,
    base_url="https://llmproxy.ai.orange"
)

pregunta = "¿Cuántas 'a' tiene la palabra MasOrange?"

resp = client.responses.create(
    model="openai/gpt-4o-mini",
    input=pregunta
)

print(resp.output_text)
