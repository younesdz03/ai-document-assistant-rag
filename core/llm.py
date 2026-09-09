import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


def get_gemini_client() -> genai.Client:
    """
    Crée et retourne un client Gemini configuré
    avec la clé API.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "La variable GEMINI_API_KEY est absente du fichier .env."
        )

    return genai.Client(api_key=api_key)


def generate_answer(
    question: str,
    context: str
) -> str:
    """
    Génère une réponse avec Gemini en utilisant
    uniquement le contexte récupéré par le système RAG.
    """

    if not question or not question.strip():
        raise ValueError("La question est vide.")

    if not context or not context.strip():
        return "Je ne trouve pas cette information dans le document."

    client = get_gemini_client()

    prompt = f"""
Tu es un assistant spécialisé dans l'analyse de documents.

Réponds à la question en utilisant uniquement les informations
présentes dans le contexte fourni.

Règles :
- N'invente aucune information.
- Si la réponse n'est pas présente dans le contexte, réponds :
  "Je ne trouve pas cette information dans le document."
- Donne une réponse claire et concise.
- Utilise la même langue que la question de l'utilisateur.

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    if not response.text:
        return "Aucune réponse n'a pu être générée."

    return response.text.strip() 

def generate_summary(
    text: str,
    language: str = "French"
) -> str:
    """
    Génère un résumé global d'un document avec Gemini.
    """

    if not text or not text.strip():
        raise ValueError("Le texte à résumer est vide.")

    client = get_gemini_client()

    prompt = f"""
Tu es un assistant spécialisé dans l'analyse de documents.

Résume le document suivant de manière claire, structurée et fidèle.

Règles :
- N'invente aucune information.
- Conserve uniquement les idées importantes.
- Organise le résumé avec des points ou de courts paragraphes.
- Utilise la langue suivante : {language}.

DOCUMENT :
{text}

RÉSUMÉ :
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    if not response.text:
        return "Aucun résumé n'a pu être généré."

    return response.text.strip()