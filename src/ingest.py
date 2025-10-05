#Ingestion de 10 PDF (chunking + embeddings → Weaviate)

#Objectif : parcourir data/pdfs/, extraire texte & métadonnées, chunker, embedder via Mistral, indexer dans Weaviate.

# Ingestion de 10 PDF (chunking + embeddings → Weaviate)
# Objectif : parcourir data/pdfs/, extraire texte & métadonnées, chunker,
# embedder via Mistral, indexer dans Weaviate.

import os
from typing import Any, Dict, Optional
from tqdm import tqdm
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from .bootstrap import bootstrap

PDF_DIR = "data/pdfs"

def _to_int_or_none(val: Any) -> Optional[int]:
    """Essaie de convertir une valeur en int. Retourne None si impossible."""
    if val is None:
        return None
    try:
        # val peut être déjà un int, une str "4", "04", etc.
        return int(str(val).strip())
    except Exception:
        return None

def _build_metadata(d) -> Dict[str, Any]:
    """Construit un dict de métadonnées compatible avec le schéma Weaviate.
    - title: TEXT
    - source: TEXT
    - page: INT (optionnel, uniquement si convertible)
    """
    title = d.metadata.get("file_name") or os.path.basename(d.metadata.get("file_path", "document.pdf"))
    source = d.metadata.get("file_path") or d.metadata.get("file_name") or ""

    # Plusieurs libs mettent la page sous des clés différentes : "page_label", "page", "page_number"...
    raw_page = (
        d.metadata.get("page")
        or d.metadata.get("page_label")
        or d.metadata.get("page_number")
        or d.metadata.get("page_index")
    )
    page_int = _to_int_or_none(raw_page)

    md = {
        "title": title,
        "source": source,
        # "page" uniquement si on a un entier, sinon on omet (évite l'erreur Weaviate)
    }
    if page_int is not None:
        md["page"] = page_int

    return md

def ingest():
    # 1) Bootstrap services (LLM/embeddings + vector store)
    client, vector_store = bootstrap()

    # 2) Charger les PDF
    if not os.path.isdir(PDF_DIR):
        raise RuntimeError(f"Dossier introuvable: {PDF_DIR}")
    docs = SimpleDirectoryReader(PDF_DIR, recursive=False).load_data()
    if len(docs) == 0:
        raise RuntimeError("Aucun PDF trouvé dans data/pdfs.")

    # 3) Chunking fin
    splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=150)
    nodes = []
    for d in tqdm(docs, desc="Chunking"):
        split_nodes = splitter.split_text(d.text)
        base_meta = _build_metadata(d)
        for i, chunk_text in enumerate(split_nodes):
            # Remplacer d.copy() (déprécié en Pydantic v2) par model_copy()
            n = d.model_copy()
            n.text = chunk_text
            # Copie défensive pour ne pas muter base_meta
            n.metadata = dict(base_meta)
            nodes.append(n)

    # 4) Construire l'index sur Weaviate
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(nodes, storage_context=storage_context, show_progress=True)

    print(f"Ingestion terminée. {len(nodes)} chunks indexés.")

if __name__ == "__main__":
    ingest()
