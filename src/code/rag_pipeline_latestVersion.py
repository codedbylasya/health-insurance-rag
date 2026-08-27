import os
import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import cohere
import weaviate
from huggingface_hub import InferenceClient, InferenceTimeoutError
from huggingface_hub.errors import HfHubHTTPError
from pii_guardrails import pii_guardrail
from groq import Groq
import groq

load_dotenv()
# ============ 1. INITIALIZATION ============
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "rerank-v3.5"
LLM_MODEL = "openai/gpt-oss-20b"
CURRENT_PROMPT_VERSION = "v12_1"
#hf_client = InferenceClient(token=os.getenv("HF_PORTFOLIO_PROJECTS_KEY"))
cohere_client = cohere.ClientV2(os.getenv("CO_PORTFOLIO_PROJECTS_KEY_V1"))
groq_client = Groq(api_key=os.getenv("GROQ_PORTFOLIO_PROJECTS_KEY")) 
def run_rag_pipeline(working_question: str) -> dict:
    """Run the RAG pipeline end-to-end for a single question.

    Retrieval uses DUAL-QUERY FUSION: one hybrid search at alpha=0.5 (the
    normal balanced behavior) and one pure-keyword search at alpha=0.0
    (catches cases where a chunk shares a distinctive keyword with the
    query but scores badly on vector similarity - see chunk 409/Cosmetic
    Services investigation). Results are merged and deduped by chunk UUID
    before reranking, so a chunk only needs to be found by EITHER method
    to reach the reranker.
    """
    pii_check = pii_guardrail(user_question=working_question)
    user_question = pii_check["redacted_question"]
    sample_query = f"Represent this sentence for searching relevant passages: {user_question}"
    query_vector = hf_client.feature_extraction(model=EMBED_MODEL, text=sample_query)

    with weaviate.connect_to_local() as c:
        collection = c.collections.get("DocumentChunk")

        # Query 1: normal balanced hybrid search (existing production behavior)
        balanced_response = collection.query.hybrid(
            query=user_question,
            vector=query_vector.tolist(),
            alpha=0.5,
            limit=20
        )

        # Query 2: pure keyword search (catches strong keyword matches that
        # score poorly in vector space and get excluded from balanced search)
        keyword_response = collection.query.hybrid(
            query=user_question,
            vector=query_vector.tolist(),
            alpha=0.0,
            limit=20
        )

        # Merge + dedupe by UUID, preserving order (balanced results first)
        seen_uuids = set()
        merged_objects = []
        for obj in list(balanced_response.objects) + list(keyword_response.objects):
            if obj.uuid not in seen_uuids:
                seen_uuids.add(obj.uuid)
                merged_objects.append(obj)
        # ============ RERANK THE MERGED POOL ============
        docs = [obj.properties.get('text') for obj in merged_objects]
        rerank_response = cohere_client.rerank(
            model=RERANK_MODEL,
            query=user_question,
            documents=docs,
            top_n=3
        )
        top_context = []
        for r in rerank_response.results:
            obj = merged_objects[r.index]
            top_context.append({
                "text": obj.properties.get('text'),
                "score": r.relevance_score,
                "pages": obj.properties.get('pages', []),   # full page range, not just first page
                "source": obj.properties.get('source_file')
            })

    # ============ BUILD LLM CONTEXT ============
    llm_context_block = ""
    for num, item in enumerate(top_context, start=1):
        pages = item['pages']
        page_str = f"{pages[0]}-{pages[-1]}" if len(pages) > 1 else str(pages[0]) if pages else "unknown"
        llm_context_block += f"[Source: {item['source']}, Page {page_str}]\n"
        llm_context_block += f"{item['text']}\n\n"

    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", f"rag_template_{CURRENT_PROMPT_VERSION}.txt")
    with open(prompt_path, "r", encoding="utf-8") as file:
        template_blueprint = file.read()

    final_prompt = template_blueprint.format(
        context=llm_context_block,
        question=user_question
    )
    print(f"\n========== QUESTION SENT TO LLM ==========\n{user_question}\n")
    answer = None
    error = None
    try:
        #completion = hf_client.chat.completions.create(
        completion = groq_client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=500,
            temperature=0,
            reasoning_effort="low",
            messages=[{"role": "user", "content": final_prompt}]
        )
        answer = completion.choices[0].message.content
        answer_pii_check = pii_guardrail(user_question=None, generated_answer=answer)
        answer = answer_pii_check["redacted_answer"]
    
    except InferenceTimeoutError:
        error = "LLM request timed out"
    except HfHubHTTPError as e:
        if "429" in str(e):
            error = "Rate limit hit — please retry in a moment"
        else:
            error = f"HF API error: {str(e)}"
    except Exception as e:
        error = f"Unexpected error: {str(e)}"
    except groq.APITimeoutError:
        error = "LLM request timed out"
    """except groq.RateLimitError as e:
        error = "Rate limit hit — please retry in a moment"
    except groq.APIConnectionError as e:
        error = f"Could not reach Groq's servers: {str(e)}"
    except groq.APIStatusError as e:
        error = f"Groq API error {e.status_code}: {str(e)}"
    except Exception as e:
        error = f"Unexpected error: {str(e)}"
        """
    return {
        "question": user_question,
        "answer": answer,
        "error": error,
        "context": top_context,
        "prompt": final_prompt,
    }

if __name__ == "__main__":
    working_question = input("Please type in your query..")
    result = run_rag_pipeline(working_question)

    if result["error"]:
        print(f"\n Cloud Connection Error: {result['error']}")
    else:
        print("\n-----------------FINAL CLOUD LLM ANSWER-----------------------\n")
        print(result["answer"])
        print("\n--------------------------------------------")
