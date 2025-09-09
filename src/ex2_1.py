from pathlib import Path
import os
from dotenv import load_dotenv
import openai

# Cargamos la api key desde nuestro .env
load_dotenv()
api_key = os.getenv("LITELLM_KEY")

client = openai.OpenAI(api_key=api_key, base_url="https://llmproxy.ai.orange")

# Declaramos la frase y consultamos el embedding
text = "You shall not pass!"

response = client.embeddings.create(
    model="openai/text-embedding-3-small",
    input=text
)

embedding = response.data[0].embedding

print("Longitud del embedding:", len(embedding))
print("Embedding:", embedding)

#Bonus: Gandalf en Moria, la duda ofende ;)