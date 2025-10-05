# checking the cluster status

import weaviate
from weaviate.classes.init import Auth
from .settings import (
    WEAVIATE_URL, WEAVIATE_API_KEY
)
print("Weaviate URL:", WEAVIATE_URL)
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
)

print(client.is_ready())  # Should print: `True`

client.close()  # Free up resources