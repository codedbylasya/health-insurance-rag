import modal
import os
import json

image = modal.Image.debian_slim().pip_install("sentence-transformers")
app = modal.App("heading-combined-embed-test")

RERANK_MODEL = "rerank-v3.5"
TEST_COLLECTION_NAME = "DocumentChunkHeadingTest"


@app.function(image=image, timeout=1800, gpu="T4")
def embed_combined_texts(combined_texts: list[str]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    embeddings = model.encode(combined_texts, show_progress_bar=True)
    return embeddings.tolist()


@app.local_entrypoint()
def main():
    import weaviate
    import cohere
    from weaviate.classes.config import Property, DataType
    from dotenv import load_dotenv
    load_dotenv()


    # ============ 1. LOAD THE CORRECTED CHUNKS (with fixed headings) ============
    chunks_path = os.path.join(os.path.dirname(__file__), "..", "data", "chunks_toc_fix_test.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    # Build heading+text combined strings for embedding (original "text" stays
    # untouched for display/retrieval-result purposes)
    combined_texts = []
    for c in chunks_data:
        heading = " > ".join(c.get("headings") or [])
        combined_texts.append(f"{heading}\n\n{c['text']}" if heading else c["text"])

    print(f"Embedding {len(combined_texts)} chunks with heading-combined text on Modal...")
    embeddings = embed_combined_texts.remote(combined_texts)
    print("Done embedding.\n")

    # ============ 2. INGEST INTO A SEPARATE TEST COLLECTION (production untouched) ============
    properties = [
        Property(name="text", data_type=DataType.TEXT),
        Property(name="page", data_type=DataType.INT),
        Property(name="pages", data_type=DataType.INT_ARRAY),
        Property(name="headings", data_type=DataType.TEXT_ARRAY),
        Property(name="content_types", data_type=DataType.TEXT_ARRAY),
        Property(name="source_file", data_type=DataType.TEXT),
    ]

    with weaviate.connect_to_local() as client:
        if client.collections.exists(TEST_COLLECTION_NAME):
            client.collections.delete(TEST_COLLECTION_NAME)
            print(f"Deleted old {TEST_COLLECTION_NAME} collection.")

        client.collections.create(name=TEST_COLLECTION_NAME, properties=properties)
        collection = client.collections.get(TEST_COLLECTION_NAME)

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
                    vector=vector,
                )
        print(f"Ingested {len(chunks_data)} chunks into {TEST_COLLECTION_NAME}.\n")

        # ============ 3. RUN THE SAME HYBRID SEARCH + RERANK LOGIC AS PRODUCTION ============
        user_question = "Is cosmetic surgery something my plan pays for?"

        from sentence_transformers import SentenceTransformer  # local, CPU is fine for one query
        query_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        sample_query = f"Represent this sentence for searching relevant passages: {user_question}"
        query_vector = query_model.encode(sample_query)

        hybrid_response = collection.query.hybrid(
            query=user_question,
            vector=query_vector.tolist(),
            alpha=0.5,
            limit=20
        )

        print("========== TOP 20 HYBRID SEARCH RESULTS (heading-combined embeddings) ==========")
        for rank, obj in enumerate(hybrid_response.objects, start=1):
            print(f"Rank {rank:2}  page {obj.properties.get('page')}  {obj.properties.get('text')[:70]!r}")

        cohere_client = cohere.ClientV2(os.getenv("CO_PORTFOLIO_PROJECTS_KEY"))
        docs = [obj.properties.get("text") for obj in hybrid_response.objects]
        rerank_response = cohere_client.rerank(
            model=RERANK_MODEL, query=user_question, documents=docs, top_n=3
        )

        print("\n========== TOP 3 AFTER RERANK ==========")
        for rank, r in enumerate(rerank_response.results, start=1):
            obj = hybrid_response.objects[r.index]
            print(f"Rank {rank}  original_hybrid_rank={r.index + 1}  score={r.relevance_score:.4f}")
            print(f"  page {obj.properties.get('page')}: {obj.properties.get('text')[:100]!r}\n")