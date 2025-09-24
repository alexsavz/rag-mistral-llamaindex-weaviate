#Ingestion de 10 PDF (chunking + embeddings → Weaviate)

#Objectif : parcourir data/pdfs/, extraire texte & métadonnées, chunker, embedder via Mistral, indexer dans Weaviate.

import os
from tqdm import tqdm
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from .bootstrap import bootstrap, close_weaviate_client

PDF_DIR = "data/pdfs"

def ingest():
    client, vector_store = bootstrap()

    try:
        if not os.path.isdir(PDF_DIR):
            raise RuntimeError(f"Dossier introuvable: {PDF_DIR}")
        docs = SimpleDirectoryReader(PDF_DIR, recursive=False).load_data()
        if len(docs) == 0:
            raise RuntimeError("Aucun PDF trouvé dans data/pdfs.")

        # 1) Produire des NODES (et non pas des Documents tronqués)
        splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=150)
        nodes = splitter.get_nodes_from_documents(docs)

        # (facultatif) Harmoniser quelques métadonnées visibles dans les citations
        for n in nodes:
            meta = n.metadata or {}
            fname = meta.get("file_name") or os.path.basename(meta.get("file_path", "document.pdf"))
            n.metadata.update({
                "title": fname,
                "source": meta.get("file_path", fname),
                "page": meta.get("page_label", meta.get("page_label", "")),
            })

        # 2) Construire l'index dans Weaviate
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        _ = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True,
        )

        print(f"Ingestion terminée. {len(nodes)} chunks indexés.")
    finally:
        # Toujours fermer la connexion
        close_weaviate_client(client)

if __name__ == "__main__":
    ingest()
