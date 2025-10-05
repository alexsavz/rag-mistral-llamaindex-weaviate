## Installer les dépendances
pip install -r requirements.txt

## Lancer Weaviate en local (Docker)
docker compose up -d

## Ingestion (une seule fois ou après ajout de nouveaux PDF)
python -m src.ingest

## Poser une question
python -m src.chat
