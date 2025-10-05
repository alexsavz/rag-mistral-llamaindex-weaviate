#Connexion LlamaIndex ↔ Mistral ↔ Weaviate
#initialise LlamaIndex (LLM + embeddings) et Weaviate

from typing import Optional
import weaviate
from weaviate.classes.config import Property, DataType, Configure
from llama_index.core import Settings
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from .settings import (
    MISTRAL_API_KEY, MISTRAL_LLM_MODEL, MISTRAL_EMBED_MODEL,
    WEAVIATE_URL, WEAVIATE_API_KEY, WEAVIATE_CLASS
)

def get_weaviate_client() -> weaviate.WeaviateClient:
    if WEAVIATE_API_KEY:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL, auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY)
        )
    else:
        # v4: helper local (host/port auto)
        client = weaviate.connect_to_local()
    return client

def close_client(client: weaviate.WeaviateClient):
    try:
        client.close()
    except Exception:
        pass

def ensure_weaviate_schema(client: weaviate.WeaviateClient, class_name: str):
    # v4: list_all() renvoie des objets; on extrait les noms
    existing = [c.name for c in client.collections.list_all()]
    if class_name in existing:
        return
    client.collections.create(
        name=class_name,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="chunk", data_type=DataType.TEXT),
            Property(name="page", data_type=DataType.INT),
        ],
    )

def configure_llamaindex():
    llm = MistralAI(api_key=MISTRAL_API_KEY, model=MISTRAL_LLM_MODEL, temperature=0.1)
    embed_model = MistralAIEmbedding(api_key=MISTRAL_API_KEY, model_name=MISTRAL_EMBED_MODEL)
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 1000
    Settings.chunk_overlap = 150

def get_vector_store(client: weaviate.WeaviateClient) -> WeaviateVectorStore:
    return WeaviateVectorStore(
        weaviate_client=client,
        index_name=WEAVIATE_CLASS,
        text_key="chunk",
    )

def bootstrap():
    configure_llamaindex()
    client = get_weaviate_client()
    ensure_weaviate_schema(client, WEAVIATE_CLASS)
    return client, get_vector_store(client)

