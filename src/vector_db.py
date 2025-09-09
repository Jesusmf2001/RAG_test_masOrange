from abc import ABC, abstractmethod
from pathlib import Path

class VectorDatabaseInterface(ABC):
    """
    Abstract interface for vector database implementations.
    
    Expected constructor signature:
        __init__(self) -> None
            Should initialize:
            - self.client: openai.OpenAI client
            - self.embeddings: List for storing embeddings
            - self.chunks: List for storing text chunks
    """
    
    @abstractmethod
    def load_document(self, markdown_text: str) -> None:
        """Load and process a document into embeddings."""
        pass
    
    @abstractmethod
    def print_number_of_embeddings(self) -> None:
        """Print the number of stored embeddings."""
        pass


class VectorDDBB(VectorDatabaseInterface):
    """Implementación de la base de datos vectorial"""

    def __init__(self, client=None):
        self.client = client
        self.chunks = [] 
        self.embeddings = []

    def load_document(self, markdown_text: str) -> None:
        """Se divide el texto por '##' y genera embeddings para cada chunk"""
        if self.client is None:
            raise ValueError("Debes pasar un cliente OpenAI para generar embeddings.")

        raw_chunks = markdown_text.split("##")
        for chunk in raw_chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            self.chunks.append(chunk)

            # Generar embedding usando OpenAI
            response = self.client.embeddings.create(
                model="openai/text-embedding-3-small",
                input=chunk
            )
            embedding = response.data[0].embedding
            self.embeddings.append(embedding)

    def print_number_of_embeddings(self) -> None:
        print(f"Número de embeddings almacenados: {len(self.embeddings)}")

    def _nearest_chunks(self, embedding: list[float], top_n: int = 3) -> list[str]:
        similarities = [(sum(a*b for a,b in zip(embedding, e)), i) for i,e in enumerate(self.embeddings)]
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [self.chunks[i] for _, i in similarities[:top_n]]

    def nearest_chunks(self, text: str) -> list[str]:
        response = self.client.embeddings.create(model="openai/text-embedding-3-small", input=text)
        embedding = response.data[0].embedding
        return self._nearest_chunks(embedding)

    def load_document_from_path(self, markdown_path: Path) -> None:
        """Load and process a document into embeddings."""
        with open(markdown_path, "r", encoding="utf-8") as f:
            self.load_document(f.read())

