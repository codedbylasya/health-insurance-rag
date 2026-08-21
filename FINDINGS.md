# Methodology, Evaluation & Findings

## Retrieval: dual-query fusion

Instead of a single hybrid search call, retrieval runs two queries per
question — a balanced hybrid search (`alpha=0.5`) and a pure-keyword search
(`alpha=0.0`) — merged and deduplicated by chunk ID before reranking.

**Why:** paraphrased questions that share few keywords with the source text
(e.g. "cosmetic surgery" vs. the document's own term "Cosmetic Services")
were being missed. A single blended `alpha` score forces a trade-off between
keyword precision and semantic recall; running both queries and taking the
union avoids that trade-off entirely.

## Evaluation

A hand-verified, 20-question golden dataset covers simple lookups,
paraphrased questions, multi-hop reasoning, near-duplicate disambiguation,
and genuinely unanswerable questions. Every fact and page number was
verified directly against the source PDF's own printed page numbers, not
assumed from chunk metadata.

**Five checks run on every eval pass:**

| Check | Method | What it catches |
|---|---|---|
| Context Recall | RAGAS (LLM-judged) | Did retrieval get everything needed to answer? |
| Context Precision | RAGAS (LLM-judged) | Was what was retrieved relevant and well-ranked? |
| Faithfulness | RAGAS (LLM-judged) | Is the answer grounded in retrieved context? |
| Factual Correctness | RAGAS (LLM-judged) | Does the answer match the reference? |
| Citation Correctness | Custom, deterministic | Does the cited page match the ground truth page(s)? |

Unanswerable questions are excluded from the Recall/Precision/Faithfulness
averages — a 0.0 score on a correctly-declined question is expected
behavior, not a failure, and would otherwise distort the aggregate signal.

**CI/CD:** a GitHub Actions workflow runs the full eval suite on every
push/PR against a disposable Weaviate service container, failing the build
if any metric drops below a threshold set with a safety margin below the
validated baseline — enough to absorb normal LLM-judge run-to-run variance
without false-failing on noise.

## A real bug found through this process

Golden dataset page numbers were originally off by one across the entire
document. Root cause: the ingestion pipeline's page-numbering came from
Docling's raw physical page count, which didn't account for an unnumbered
cover page — so every citation was silently pointing to the wrong page
relative to the document's own printed footer numbers. Fixed at the source
(`process_documents.py`) and re-verified across the full dataset by
building a reliable footer-to-page map and re-checking every fact directly,
rather than trusting chunk metadata.

## Guardrails

Built after confirming, through repeated testing, that prompt instructions
alone are not a reliable defense against certain failure modes — a model
told explicitly and repeatedly not to do something can still do it under
complex multi-source synthesis. This mirrors current OWASP guidance for LLM
applications: harden the surrounding system so failures are contained,
rather than relying on the model to never fail.

**Active in the live pipeline:**

- **PII detection + redaction** (Microsoft Presidio) — scans and redacts
  personal information from both the question and the generated answer,
  before either is used further. Forward-looking: the current source
  document is a generic template with no real PHI; this exists for when
  real, user-uploaded EOC documents (which typically carry a real name and
  member ID) are supported.

**Built, tested, not yet wired into the live pipeline (v2 roadmap):**

- **Citation grounding check** — deterministic check confirming every cited
  page was actually part of the retrieved context, catching citations to
  pages the model never saw. Currently used as part of the eval suite;
  not yet run as a live, per-query guardrail.
- **Prompt injection detection** — pattern-based check for common
  instruction-override attempts in user input. Built and tested standalone,
  not yet integrated.

## Known limitations

- **A reproducible generation-model failure mode:** when several retrieved
  chunks discuss related-but-distinct subjects (e.g. two differently-scoped
  exclusion clauses on the same page), the model can misattribute a claim
  to the wrong source — even when the correct source is present in context
  and ranked highest by the reranker. Confirmed reproducible across
  multiple prompt versions and two different generation models; this
  appears to be a genuine model limitation, not a prompt or retrieval gap.
- **Free-tier API quotas** (HuggingFace, Groq) are small enough to be hit
  during normal development. The project currently switches between
  providers depending on availability.
- **No multi-document/multi-user support yet** — the system is scoped to
  one fixed document. This is a deliberate, stated scope boundary.