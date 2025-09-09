import os
from pathlib import Path
from dotenv import load_dotenv
import openai
from vector_db import VectorDDBB

if __name__ == "__main__":
    load_dotenv()
    client = openai.OpenAI(api_key=os.getenv("LITELLM_KEY"), base_url="https://llmproxy.ai.orange")

    vector_db = VectorDDBB(client)
    vector_db.load_document_from_path(Path("../ai-engineer-evaluation-test.md"))
    vector_db.print_number_of_embeddings()

    nearest_chunk = vector_db.nearest_chunks("Darle funcionalidad a la base de datos")[0]
    print("\nChunk más parecido:\n", nearest_chunk)
