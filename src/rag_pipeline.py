import os
import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import cohere
import weaviate

load_dotenv()

# ============ 1. INITIALIZATION ============
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "rerank-v3.5"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CURRENT_PROMPT_VERSION = "v3_0"

#===============API CLIENTS==================================
hf_client = InferenceClient(token = os.getenv("HF_PORTFOLIO_PROJECTS_KEY"))
cohere_client = cohere.ClientV2(os.getenv("CO_PORTFOLIO_PROJECTS_KEY"))

llm_context_block = ""
user_question = "If I receive emergency services from an out-of-network provider, will I be charged out-of-network cost shares?"

# ============ 3. EMBED THE QUERY ============
sample_query = f"Represent this sentence for searching relevant passages: {user_question}"
query_vector = hf_client.feature_extraction(model = EMBED_MODEL, text=sample_query)
# ============ 4. WEAVIATE HYBRID SEARCH ============
with weaviate.connect_to_local() as c:
    collection = c.collections.get("DocumentChunk")
    hybrid_response = collection.query.hybrid(
        query=user_question,
        vector=query_vector.tolist(),
        alpha=0.5,
        limit=10
    )
# ============ 5. RERANK ============
docs = [obj.properties.get('text') for obj in hybrid_response.objects]
response = cohere_client.rerank(
    model= RERANK_MODEL,
    query=user_question,
    documents=docs,
    top_n=3
)
top_3_context = []
for r in response.results:
    obj = hybrid_response.objects[r.index]
    top_3_context.append({
        "text": obj.properties.get('text'),
        "score": r.relevance_score,
        "page_number": obj.properties.get('page'),
        "source": obj.properties.get('source_file')
    })
# ============ 7. BUILD LLM CONTEXT ============
for num, item in enumerate(top_3_context, start=1):
    llm_context_block += f"(--------Document Chunk #{num} Source: {item['source']})(PDF Page: {item['page_number']}) ---\n"
    llm_context_block += f"{item['text']}\n\n"

prompt_path = f"prompts/rag_template_{CURRENT_PROMPT_VERSION}.txt"
with open(prompt_path, "r",encoding ="utf-8") as file:
    template_blueprint =file.read()

final_prompt = template_blueprint.format(
    context=llm_context_block,
    question=user_question
)
# ============ 8. LLM CALL ============
try:
    completion = hf_client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=500,
        messages=[
            {"role": "user", "content": final_prompt}
        ]
    )
    print("\n-----------------FINAL CLOUD LLM ANSWER-----------------------\n")
    print(completion.choices[0].message.content)
    print("\n--------------------------------------------")

except Exception as e:
    print(f"\n Cloud Connection Error: {e}")


"""# ============ 9. OPTIONAL: DISPLAY TOP 3 CHUNKS ============

print("\n================ TOP 3 FINAL CHUNKS WITH PAGES ================")
for num, item in enumerate(top_3_context, start=1):
    print(f"\nFinal Context Chunk #{num} [Source Document: Page {item['page_number']}]")
    print(item['text'])
    print("─" * 40)"""