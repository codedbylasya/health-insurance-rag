"""
Citation correctness check.

RAGAS has no built-in metric for this, so it's a deterministic, non-LLM
check: extract the page number(s) the pipeline actually cited from its
response text, and compare against the golden dataset's ground_truth_page.

A row is marked correct if the ground_truth_page appears ANYWHERE in the
set of pages the model cited (not requiring an exact single match, since
some correct answers legitimately cite multiple pages).

Unanswerable rows (ground_truth_page is None) are excluded from this
check entirely -- there's nothing to cite correctly or incorrectly.
"""

import re
import json
import os


def extract_cited_pages(response_text: str) -> list[int]:
    """
    Pulls every page number out of citation patterns like:
    (Source: Anthem_EOC.pdf, Page 24)
    (Anthem_EOC.pdf, Page 24, 87)
    (Source: Anthem_EOC.pdf, Page 10-11)   <- range, from multi-page chunks
    Returns a list of ints, e.g. [24] or [24, 87] or [10, 11]. Empty list if none found.
    """
    if not response_text:
        return []

    # matches "Page 24" / "Page 24, 87" / "Page 10-11" / "Page 10 - 11", case-insensitive
    matches = re.findall(r"Page\s+(\d+(?:\s*[-,]\s*\d+)*)", response_text, re.IGNORECASE)

    pages = []
    for m in matches:
        # a single match could be "24", "24, 87", or "10-11" -- split on both separators
        parts = re.split(r"[-,]", m)
        for num in parts:
            num = num.strip()
            if num.isdigit():
                pages.append(int(num))
    return pages


def check_citation_correctness(golden_dataset: list[dict], pipeline_outputs: list[dict]) -> list[dict]:
    """
    golden_dataset: list of rows from golden_dataset.json (has ground_truth_page, category)
    pipeline_outputs: list of rows from pipeline_results.json (has 'answer')

    Returns a list of per-row results, and prints a summary.
    """
    results = []

    for golden_row, outcome in zip(golden_dataset, pipeline_outputs):
        if golden_row["category"] == "unanswerable" or not golden_row.get("ground_truth_pages"):
            results.append({
                "id": golden_row["id"],
                "question": golden_row["question"],
                "category": golden_row["category"],
                "citation_correct": None,  # not applicable
                "cited_pages": [],
                "ground_truth_pages": [],
            })
            continue

        response = outcome.get("answer", "") or ""
        cited_pages = extract_cited_pages(response)
        ground_truth_pages = golden_row["ground_truth_pages"]

        # correct if ANY cited page matches ANY valid ground truth page
        is_correct = any(p in cited_pages for p in ground_truth_pages)

        results.append({
            "id": golden_row["id"],
            "question": golden_row["question"],
            "category": golden_row["category"],
            "citation_correct": is_correct,
            "cited_pages": cited_pages,
            "ground_truth_pages": ground_truth_pages,
        })

    return results


def summarize(results: list[dict]):
    scorable = [r for r in results if r["citation_correct"] is not None]
    correct = [r for r in scorable if r["citation_correct"]]
    incorrect = [r for r in scorable if not r["citation_correct"]]

    print("=== Citation Correctness ===")
    print(f"Scorable questions (excludes unanswerable): {len(scorable)}")
    if scorable:
        print(f"Correct: {len(correct)}  ({len(correct)/len(scorable):.1%})")
        print(f"Incorrect: {len(incorrect)}  ({len(incorrect)/len(scorable):.1%})")

    if incorrect:
        print("\n--- Incorrect citations ---")
        for r in incorrect:
            print(f"{r['id']}: expected page(s) {r['ground_truth_pages']}, "
                  f"cited {r['cited_pages'] or '(no citation found)'}")
            print(f"   Q: {r['question']}")


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")

    with open(os.path.join(data_dir, "golden_dataset.json"), "r", encoding="utf-8") as f:
        golden_dataset = json.load(f)

    with open(os.path.join(data_dir, "pipeline_results.json"), "r", encoding="utf-8") as f:
        pipeline_outputs = json.load(f)

    results = check_citation_correctness(golden_dataset, pipeline_outputs)
    summarize(results)

    # save detailed results alongside your other eval outputs
    out_path = os.path.join(data_dir, "citation_correctness_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {out_path}")