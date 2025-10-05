# Vérifier la présence de la collection & son nombre d’objets

from .bootstrap import get_weaviate_client, close_client
from .settings import WEAVIATE_CLASS

def main():
    client = get_weaviate_client()
    try:
        # Lister les collections
        cols = client.collections.list_all()
        print("Collections:", [c.name for c in cols])

        # Récupérer la collection et compter les objets
        col = client.collections.get(WEAVIATE_CLASS)
        count = col.aggregate.over_all(total_count=True).total_count
        print(f"Collection {WEAVIATE_CLASS} → objets = {count}")
    finally:
        close_client(client)

if __name__ == "__main__":
    main()
