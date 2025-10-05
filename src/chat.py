#Chat / Q&A avec citations des sources

#Nous configurons un Query Engine LlamaIndex qui va :

#récupérer les k chunks pertinents dans Weaviate,

#demander au LLM Mistral une réponse synthétique basée sur ces extraits,

#renvoyer la liste des sources (titre + page + chemin).

from llama_index.core import VectorStoreIndex
from llama_index.core.response.notebook_utils import display_response
from .bootstrap import bootstrap

def make_query_engine():
    client, vector_store = bootstrap()
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index.as_query_engine(
        similarity_top_k=4,
        response_mode="compact",
    )

def ask(query: str):
    qe = make_query_engine()
    response = qe.query(query)

    # Affichage console + citations
    print("\n=== RÉPONSE ===\n")
    print(str(response))
    print("\n=== SOURCES ===\n")
    for i, s in enumerate(response.source_nodes, start=1):
        md = s.metadata or {}
        title = md.get("title", "Document")
        page = md.get("page", "?")
        source = md.get("source", "")
        print(f"[{i}] {title} (page {page}) - {source}")

if __name__ == "__main__":
    # Exemple de question ciblée
    q = "Quelles sont les recommandations (HAS ou ouvrages) pour la rééducation du LCA entre S4 et S8 ?"
    ask(q)

