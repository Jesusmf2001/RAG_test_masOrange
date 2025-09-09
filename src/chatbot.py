from vector_db import VectorDDBB

class Chatbot:
    def __init__(self, vector_db: VectorDDBB, client, top_k: int = 3):
        self.vector_db = vector_db
        self.client = client
        self.top_k = top_k

    def ask_question(self, question: str) -> str:
        # Se obtienen los chunks mas cercanos
        context_chunks = self.vector_db.nearest_chunks(question)[:self.top_k]
        context_text = "\n\n".join(context_chunks)

        # Se prepara el
        prompt = f"""
        Eres un asistente que responde preguntas usando el contexto proporcionado.
        Contexto:
        {context_text}

        Pregunta:
        {question}
        """
        response = self.client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content.strip()
