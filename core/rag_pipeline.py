from typing import List, Dict

from core.vector_store import VectorStore
from core.retriever import Retriever
from core.llm import generate_answer





class RAGPipeline:
    """
    Pipeline RAG complet :
    - stockage des chunks
    - recherche vectorielle
    - construction du contexte
    - génération de réponse avec Gemini
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "documents"
    ):
        self.vector_store = VectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name
        )

        self.retriever = Retriever(
            vector_store=self.vector_store
        )


    def index_chunks(
        self,
        chunks: List[Dict]
    ) -> None:
        """
        Indexe les chunks dans ChromaDB.
        """

        if not chunks:
            raise ValueError(
                "Aucun chunk à indexer."
            )

        self.vector_store.add_chunks(chunks)


    def ask(
        self,
        question: str,
        n_results: int = 3
    ) -> Dict:
        """
        Exécute le pipeline RAG complet pour une question.

        Retourne :
        - answer
        - sources
        """

        if not question or not question.strip():
            raise ValueError(
                "La question est vide."
            )

        # Récupérer les chunks pertinents
        retrieved_chunks = self.retriever.retrieve(
            query=question,
            n_results=n_results
        )

        if not retrieved_chunks:
            return {
                "answer": "Je ne trouve pas cette information dans le document.",
                "sources": []
            }

        # Construire le contexte
        context_parts = []

        for chunk in retrieved_chunks:
            context_parts.append(
                f"""
Source: {chunk['filename']}
Page: {chunk['page']}

{chunk['text']}
""".strip()
            )

        context = "\n\n---\n\n".join(context_parts)

        # Générer la réponse avec Gemini
        answer = generate_answer(
            question=question,
            context=context
        )

        # Préparer les sources à afficher dans l'interface
        sources = [
            {
                "filename": chunk["filename"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "distance": chunk["distance"]
            }
            for chunk in retrieved_chunks
        ]

        return {
            "answer": answer,
            "sources": sources
        }


    def count_documents(self) -> int:
        """
        Retourne le nombre de chunks présents dans ChromaDB.
        """

        return self.vector_store.count()


    def clear_index(self) -> None:
        """
        Vide la collection ChromaDB.
        """

        self.vector_store.clear()
