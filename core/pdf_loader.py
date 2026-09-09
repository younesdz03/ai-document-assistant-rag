from pypdf import PdfReader
from typing import List, Dict


def load_pdf(uploaded_file) -> List[Dict]:
    """
    Charge un PDF et extrait le texte page par page.

    Chaque page conserve :
    - le nom du fichier
    - le numéro de page
    - le texte extrait
    """

    pdf = PdfReader(uploaded_file)

    pages = []

    # Récupérer le nom du fichier
    filename = getattr(uploaded_file, "name", "document.pdf")

    for page_number, page in enumerate(pdf.pages, start=1):

        text = page.extract_text()

        if text and text.strip():

            pages.append({
                "filename": filename,
                "page": page_number,
                "text": text.strip()
            })

    return pages