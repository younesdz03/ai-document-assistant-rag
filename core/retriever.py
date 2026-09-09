from typing import List, Dict

from core.vector_store import VectorStore


class Retriever:
    """
    Récupère les chunks les plus pertinents depuis ChromaDB.
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store


    def retrieve(
        self,
        query: str,
        n_results: int = 3
    ) -> List[Dict]:
        """
        Retourne une liste de chunks pertinents avec leurs métadonnées.
        """

        results = self.vector_store.search(
            query=query,
            n_results=n_results
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved_chunks = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):
            retrieved_chunks.append(
                {
                    "text": document,
                    "filename": metadata.get("filename"),
                    "page": metadata.get("page"),
                    "chunk_id": metadata.get("chunk_id"),
                    "distance": distance
                }
            )

        return retrieved_chunks


    def build_context(
        self,
        query: str,
        n_results: int = 3
    ) -> str:
        """
        Construit un contexte texte prêt à être envoyé au LLM.
        """

        chunks = self.retrieve(
            query=query,
            n_results=n_results
        )

        if not chunks:
            return ""

        context_parts = []

        for chunk in chunks:
            context_parts.append(
                f"""
Source: {chunk['filename']}
Page: {chunk['page']}

{chunk['text']}
""".strip()
            )

        return "\n\n---\n\n".join(context_parts)