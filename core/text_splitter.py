from typing import List, Dict


def create_chunks(
    pages: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict]:
    """
    Découpe les pages en chunks avec chevauchement.

    Chaque chunk conserve :
    - le nom du fichier
    - le numéro de page
    - l'identifiant du chunk
    - le texte
    """

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap doit être inférieur à chunk_size."
        )

    chunks = []
    chunk_index = 0
    step = chunk_size - chunk_overlap

    for page in pages:
        text = page.get("text", "").strip()
        page_number = page.get("page")
        filename = page.get("filename", "document.pdf")

        if not text:
            continue

        for start in range(0, len(text), step):
            chunk_text = text[start:start + chunk_size].strip()

            if not chunk_text:
                continue

            chunks.append(
                {
                    "chunk_id": f"{filename}_page_{page_number}_chunk_{chunk_index}",
                    "filename": filename,
                    "page": page_number,
                    "text": chunk_text
                }
            )

            chunk_index += 1

            if start + chunk_size >= len(text):
                break

    return chunks