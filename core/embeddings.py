from typing import List

from sentence_transformers import SentenceTransformer


# Modèle léger, rapide et très utilisé pour la recherche sémantique
_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)


def generate_embedding(text: str) -> List[float]:
    """
    Génère un embedding local pour un texte unique.
    """

    if not text or not text.strip():
        raise ValueError(
            "Le texte à convertir en embedding est vide."
        )

    embedding = _model.encode(
        text.strip(),
        normalize_embeddings=True
    )

    return embedding.tolist()


def generate_embeddings(
    texts: List[str]
) -> List[List[float]]:
    """
    Génère les embeddings locaux d'une liste de textes.
    """

    clean_texts = [
        text.strip()
        for text in texts
        if text and text.strip()
    ]

    if not clean_texts:
        return []

    embeddings = _model.encode(
        clean_texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.tolist()