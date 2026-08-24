import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(__file__))
from rag_pipeline_latestVersion import run_rag_pipeline
from ragas import EvaluationDataset, evaluate
from ragas.metrics import LLMContextRecall, ContextPrecision, Faithfulness, FactualCorrectness
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from citation_correctness import check_citation_correctness, summarize

nvidia_llm = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NV_PORTFOLIO_PROJECTS_KEY"),
    model="openai/gpt-oss-120b"
)
evaluation_llm = LangchainLLMWrapper(nvidia_llm)

def collect_pipeline_results(golden_dataset):
    results = []
    for row in golden_dataset:
        print(f"Running: {row['id']} - {row['question'][:50]}...")
        outcome = run_rag_pipeline(row["question"])
        results.append(outcome)
        print(f"  Done. Answer: {outcome['answer'][:80] if outcome['answer'] else outcome['error']}")
        time.sleep(20)
    return results


def ragas_structure(golden_dataset, output_from_results):
    raga_rows = []
    for row, outcome in zip(golden_dataset, output_from_results):
        raga_rows.append({
            "user_input": row["question"],
            "retrieved_contexts": [item["text"] for item in outcome["context"]],
            "reference": row["ground_truth_answer"],
            "response": outcome["answer"],
        })
    return raga_rows
    
def run_retrieval_eval(evaluation_dataset):
    results = evaluate(
        dataset=evaluation_dataset,
        metrics=[LLMContextRecall(), ContextPrecision(), Faithfulness(), FactualCorrectness()],
        llm=evaluation_llm,
        run_config=RunConfig(max_workers=5, timeout=600)
    )
    return results

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    golden_dataset_path = os.path.join(data_dir, "golden_dataset.json")
    with open(golden_dataset_path, "r", encoding="utf-8") as file:
        golden_dataset = json.load(file)
    results_path = os.path.join(data_dir, "pipeline_results.json")
    if os.path.exists(results_path):
        print("Loading saved pipeline results (skipping pipeline re-run)...")
        with open(results_path, "r", encoding="utf-8") as file:
            output = json.load(file)
    else:
        output = collect_pipeline_results(golden_dataset)
        with open(results_path, "w", encoding="utf-8") as file:
            json.dump(output, file, indent=2)

    ragas_rows = ragas_structure(golden_dataset, output)
    evaluation_dataset = EvaluationDataset.from_list(ragas_rows)
    results = run_retrieval_eval(evaluation_dataset)
    print(results)
    df = results.to_pandas()
    df["category"] = [row["category"] for row in golden_dataset]
    answerable_df = df[df["category"] != "unanswerable"]
    print(answerable_df[["context_recall", "context_precision", "faithfulness"]].mean())
    print(df[["user_input", "context_recall", "context_precision", "faithfulness"]])
    results_csv_path = os.path.join(data_dir, "eval_results.csv")
    df.to_csv(results_csv_path, index=False)
    print(f"Saved results to {results_csv_path}")

    # --- Citation correctness (deterministic, no LLM/API calls needed) ---
    citation_results = check_citation_correctness(golden_dataset, output)
    summarize(citation_results)

    citation_json_path = os.path.join(data_dir, "citation_correctness_results.json")
    with open(citation_json_path, "w", encoding="utf-8") as f:
        json.dump(citation_results, f, indent=2)
    print(f"Saved citation results to {citation_json_path}")
