# Connexion LlamaIndex ↔ Mistral ↔ Weaviate
# définir le LLM Mistral (chat/inference) + embeddings Mistral,
# créer un schema dans Weaviate (class MedicalDoc sans vectorizer),
# préparer les Settings LlamaIndex.

import os
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_LLM_MODEL = os.getenv("MISTRAL_LLM_MODEL", "mistral-large-latest")
MISTRAL_EMBED_MODEL = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
WEAVIATE_CLASS = os.getenv("WEAVIATE_CLASS", "MedicalDoc")