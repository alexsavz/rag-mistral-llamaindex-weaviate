#Forcer une requête de similarité brute (sanity retrieval)

from llama_index.core import VectorStoreIndex
from .bootstrap import bootstrap, close_client

if __name__ == "__main__":
    client, vs = bootstrap()
    try:
        index = VectorStoreIndex.from_vector_store(vs)
        qe = index.as_query_engine(similarity_top_k=5)
        r = qe.query("Donne-moi les points clés des recommandations HAS liées à la lombalgie chronique.")
        print("Réponse:", str(r))
        print("Nb sources:", len(getattr(r, "source_nodes", []) or []))
    finally:
        close_client(client)
