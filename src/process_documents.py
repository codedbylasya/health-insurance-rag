import modal

image = modal.Image.debian_slim().apt_install("libgl1", "libglib2.0-0").pip_install("docling")
embed_image = modal.Image.debian_slim().pip_install("sentence-transformers")

app = modal.App("docling-app")


@app.function(image=image, timeout=1800, gpu="T4")
def process_document_in_cloud(file_bytes: bytes, filename: str) -> dict:
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker
    import tempfile

    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    open(tmp_path, "wb").write(file_bytes)

    result = DocumentConverter().convert(tmp_path)
    chunker = HybridChunker(tokenizer="BAAI/bge-large-en-v1.5", max_tokens=512)
    chunks = list(chunker.chunk(result.document))

    chunks_data = []
    for i, chunk in enumerate(chunks):
        pages = sorted({prov.page_no for item in chunk.meta.doc_items for prov in item.prov})
        content_types = sorted({item.label.value for item in chunk.meta.doc_items})
        chunks_data.append({
            "id": i,
            "text": chunk.text,
            "headings": chunk.meta.headings or [],
            "page": pages[0] if pages else None,
            "pages": pages,
            "content_types": content_types,
            "source_file": filename,
        })

    return {"filename": filename, "chunks": chunks_data}


@app.function(image=embed_image, timeout=1800, gpu="T4")
def embed_chunks_in_cloud(chunks_data: list[dict]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    texts = [c["text"] for c in chunks_data]
    embeddings = model.encode(texts)
    return embeddings.tolist()


@app.local_entrypoint()
def main():
    import json
    import numpy as np

    pdf_path = r"C:\Projects\PortfolioProject\health_Doc_Summarizer\data\Anthem_EOC.pdf"
    file_bytes = open(pdf_path, "rb").read()

    result = process_document_in_cloud.remote(file_bytes, "Anthem_EOC.pdf")
    chunks_data = result["chunks"]
    print(f"Got {len(chunks_data)} chunks")

    embeddings = embed_chunks_in_cloud.remote(chunks_data)
    print(f"Embedded {len(chunks_data)} chunks")

    chunks_path = r"C:\Projects\PortfolioProject\health_Doc_Summarizer\data\chunks.json"
    embeddings_path = r"C:\Projects\PortfolioProject\health_Doc_Summarizer\data\embeddings.npy"

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2)
    np.save(embeddings_path, np.array(embeddings))
    print("Saved chunks.json and embeddings.npy")