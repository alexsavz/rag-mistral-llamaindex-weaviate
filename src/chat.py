#Chat / Q&A avec citations des sources

#Nous configurons un Query Engine LlamaIndex qui va :

#récupérer les k chunks pertinents dans Weaviate,

#demander au LLM Mistral une réponse synthétique basée sur ces extraits,

#renvoyer la liste des sources (titre + page + chemin).

from llama_index.core import VectorStoreIndex
from .bootstrap import bootstrap, close_client

def make_query_engine():
    client, vector_store = bootstrap()
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index.as_query_engine(
        similarity_top_k=4,
        response_mode="compact",
    )

def ask(query: str):
    client, qe = make_query_engine()
    try:
        response = qe.query(query)
        print("\n=== RÉPONSE ===\n")
        # Gestion cas vide :
        txt = (str(response) or "").strip()
        print(txt if txt else "[Aucune réponse exploitable. Vérifie l’ingestion et/ou les sources.]")

        print("\n=== SOURCES ===\n")
        if getattr(response, "source_nodes", None):
            for i, s in enumerate(response.source_nodes, start=1):
                md = s.metadata or {}
                title = md.get("title", "Document")
                page = md.get("page", "?")
                source = md.get("source", "")
                print(f"[{i}] {title} (page {page}) - {source}")
        else:
            print("[Aucune source renvoyée]")
    finally:
        close_client(client)

if __name__ == "__main__":
    # Exemple de question ciblée
    q = "Quelles sont les recommandations (HAS ou ouvrages) pour la rééducation de la lombalgie chronique ?"
    ask(q)
