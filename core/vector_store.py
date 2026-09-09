import chromadb
from typing import List, Dict

from core.embeddings import generate_embeddings


class VectorStore:
    """
    Gère le stockage des chunks et embeddings dans ChromaDB.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "documents"
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )


    def add_chunks(self, chunks: List[Dict]) -> None:
        """
        Génère les embeddings Gemini et stocke les chunks
        dans ChromaDB avec leurs métadonnées.
        """

        if not chunks:
            return

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = generate_embeddings(texts)

        ids = [
            chunk["chunk_id"]
            for chunk in chunks
        ]

        metadatas = [
            {
                "filename": chunk["filename"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"]
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )


    def search(
        self,
        query: str,
        n_results: int = 3
    ) -> Dict:
        """
        Recherche les chunks les plus pertinents.
        """

        if not query or not query.strip():
            raise ValueError(
                "La requête de recherche est vide."
            )

        query_embedding = generate_embeddings(
            [query]
        )[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        return results


    def count(self) -> int:
        """
        Retourne le nombre de chunks stockés.
        """

        return self.collection.count()


    def clear(self) -> None:
        """
        Supprime tous les éléments de la collection.
        """

        existing = self.collection.get()

        ids = existing.get("ids", [])

        if ids:
            self.collection.delete(ids=ids)