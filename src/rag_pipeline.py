import os
from dotenv import load_dotenv
#from groq import Groq
#from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import InferenceClient
import weaviate
load_dotenv()
# 1. Initialization
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
reranker = CrossEncoder("mixedbread-ai/mxbai-rerank-base-v1")
pairs = []
final_results = []
llm_context_block = ""
system_instructions = """You are a precise document auditing assistant. 
Your sole task is to answer the user's question using ONLY the text segments provided below.
CRITICAL CONSTRAINTS:
1. Grounding: You are strictly forbidden from using any outside knowledge. If the text does not contain the answer, you fail.
2. Refusal Rule: If the provided chunks do not explicitly answer the question, state exactly: 'I cannot find the answer in the provided documents.'
3. Citation Rule: Do not say 'According to Document Chunk' or mention 'Chunks' in your answer. Write a natural, direct sentence answering the question. At the very end of your answer, cleanly add the source citations in parentheses like this: (Source: file_name.pdf, Page X)."""
user_question = "What is the maximum dollar amount covered for travel expenses per bariatric surgery?"
sample_query = f"Represent this sentence for searching relevant passages: {user_question}"
query_vector = model.encode(sample_query)

# 3. Query the Weaviate container
with weaviate.connect_to_local() as c:
    collection = c.collections.get("DocumentChunk")
    hybrid_response = collection.query.hybrid(
        query=user_question,
        vector=query_vector.tolist(),
        alpha=0.5,
        limit=10
    )

# 4. First For-Loop: Prepare text pairs for the Reranker
for obj in hybrid_response.objects:
    chunk_text = obj.properties.get('text')
    pairs.append([user_question, chunk_text])
scores = reranker.predict(pairs) # 5. Calculate attention scores

# 6. Second For-Loop: Safely unpack string metadata and pair with scores
for idx, obj in enumerate(hybrid_response.objects):    
    final_results.append({
        "text": obj.properties.get('text'),
        "rerank_score": float(scores[idx]),
        "page_number": obj.properties.get('page'),
        "source": obj.properties.get('source_file')
    })
sorted_results = sorted(final_results, key=lambda x: x["rerank_score"], reverse=True)   # 7. Sort by highest score first
top_3_context = sorted_results[:3]        # 8. Keep only the top 3 clean chunks

for num, item in enumerate(top_3_context, start=1):                       # 2. Start a loop to read your 3 chunks one by one
    llm_context_block += f"(--------Document Chunk #{num} Source: {item['source']})(PDF Page: {item['page_number']}) ---\n"
    llm_context_block += f"{item['text']}\n\n"
    
# ================= STEP 3: CONSTRUCT THE DYNAMIC USER PROMPT =================
final_user_prompt = f"Here are the document chunks:\n{llm_context_block}\n\nQuestion: {user_question}"
client = Groq(api_key=os.getenv("HF_PORTFOLIO_PROJECTS_KEY"))
try:
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        max_tokens=500,  
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": final_user_prompt}
        ]
    )

    print("\n-----------------FINAL CLOUD LLM ANSWER-----------------------\n")
    print(completion.choices[0].message.content)
    print("\n--------------------------------------------")

except Exception as e:
    print(f"\n Cloud Connection Error: {e}")

# 9. Display the results with true PDF page numbers
"""print("\n================ TOP 3 FINAL CHUNKS WITH PAGES ================")
for num, item in enumerate(top_3_context, start=1):
    print(f"\nFinal Context Chunk #{num} [Source Document: Page {item['page_number']}]")
    print(item['text'])
    print("─" * 40)"""



