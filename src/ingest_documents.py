import json
import numpy as np
import weaviate
from weaviate.classes.config import Property, DataType
# 1. Load the text chunks file
with open(r"C:\Projects\PortfolioProject\health_Doc_Summarizer\data\chunks.json", "r", encoding="utf-8") as f:
    chunks_data = json.load(f)
# 2. Load the mathematical vectors file
embeddings = np.load(r"C:\Projects\PortfolioProject\health_Doc_Summarizer\data\embeddings.npy")

print(f"Step 1 Complete! Loaded {len(chunks_data)} text chunks and {len(embeddings)} vectors.")
# 3. Define schema separately, for readability and reuse
properties = [
    Property(name="text", data_type=DataType.TEXT),
    Property(name="page", data_type=DataType.INT),
    Property(name="pages", data_type=DataType.INT_ARRAY),
    Property(name="headings", data_type=DataType.TEXT_ARRAY),
    Property(name="content_types", data_type=DataType.TEXT_ARRAY),
    Property(name="source_file", data_type=DataType.TEXT),
]

with weaviate.connect_to_local() as client:
    if client.collections.exists("DocumentChunk"):
        client.collections.delete("DocumentChunk")
        print("Deleted old DocumentChunk collection.")

    client.collections.create(name="DocumentChunk", properties=properties)
    print("Created DocumentChunk collection with new schema.")
    collection = client.collections.get("DocumentChunk")
    
    # 4. Insert all chunks + their embeddings
    with collection.batch.dynamic() as batch:
        for chunk, vector in zip(chunks_data, embeddings):
            batch.add_object(
                properties={
                    "text": chunk.get("text"),
                    "page": chunk.get("page"),
                    "pages": chunk.get("pages", []),
                    "headings": chunk.get("headings", []),
                    "content_types": chunk.get("content_types", []),
                    "source_file": chunk.get("source_file"),
                },
                vector=vector.tolist(),
            )