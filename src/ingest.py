#Ingestion de 10 PDF (chunking + embeddings → Weaviate)

#Objectif : parcourir data/pdfs/, extraire texte & métadonnées, chunker, embedder via Mistral, indexer dans Weaviate.

import os
from tqdm import tqdm
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from .bootstrap import bootstrap

PDF_DIR = "data/pdfs"

def ingest():
    # 1) Bootstrap services
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
        for i, chunk_text in enumerate(split_nodes):
            n = d.copy()
            n.text = chunk_text
            n.metadata = {
                "title": d.metadata.get("file_name", os.path.basename(d.metadata.get("file_path","document.pdf"))),
                "source": d.metadata.get("file_path", d.metadata.get("file_name","")),
                "page": d.metadata.get("page_label", d.metadata.get("page_label",""))
            }
            nodes.append(n)

    # 4) Construire l'index sur Weaviate
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(nodes, storage_context=storage_context, show_progress=True)

    print(f"Ingestion terminée. {len(nodes)} chunks indexés.")

if __name__ == "__main__":
    ingest()