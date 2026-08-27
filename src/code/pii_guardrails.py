"""
PII detection AND redaction guardrail.

Uses Microsoft Presidio (the industry-standard open-source PII detection
library) to scan text for personally identifiable information, and
actually REDACT it -- not just detect and report -- before the text
proceeds any further in the pipeline (embedding, retrieval, LLM call).

This project currently processes a generic, non-personalized EOC template
-- no real PHI is present today. This guardrail exists as forward-looking
protection: the moment real users upload their own, personalized EOC
documents (which typically include name, member ID, etc.), that content
would legally qualify as PHI under HIPAA, and this is the layer that
needs to sit in front of any such data before it reaches an LLM API that
isn't covered by a Business Associate Agreement (BAA).

Behavior: DETECT, LOG, AND REDACT. The redacted (placeholder-substituted)
text is what actually gets embedded and sent to the LLM -- the original,
unredacted text never leaves this function. This is enforcement, not
just monitoring.
"""

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Loading these is expensive (loads the spaCy NER model), so they're
# created once at import time and reused across calls.
_analyzer = AnalyzerEngine()
_anonymizer = AnonymizerEngine()

# Entity types worth flagging for this project's domain. Excludes very
# noisy/low-precision types (like generic URL, DATE_TIME) that would
# produce too many false positives on ordinary insurance-document text.
RELEVANT_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
    "CREDIT_CARD",
    "US_BANK_NUMBER",
    "LOCATION",
]


def check_for_pii(text: str, min_score: float = 0.5) -> list[dict]:
    """
    Scans text for PII. Returns a list of findings, each with the entity
    type, the actual matched text, and confidence score. Empty list if
    nothing found above the threshold.
    """
    if not text:
        return []

    results = _analyzer.analyze(
        text=text,
        language="en",
        entities=RELEVANT_ENTITIES,
        score_threshold=min_score,
    )

    findings = []
    for r in results:
        findings.append({
            "entity_type": r.entity_type,
            "matched_text": text[r.start:r.end],
            "score": round(r.score, 2),
        })
    return findings


def redact_pii(text: str, min_score: float = 0.5) -> str:
    """
    Detects and REDACTS PII in one step. Returns the text with any
    detected PII replaced by a type placeholder (e.g. "<PERSON>",
    "<US_SSN>"). If nothing is detected, returns the original text
    unchanged.
    """
    if not text:
        return text

    results = _analyzer.analyze(
        text=text,
        language="en",
        entities=RELEVANT_ENTITIES,
        score_threshold=min_score,
    )

    if not results:
        return text

    anonymized = _anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text


def pii_guardrail(user_question: str, generated_answer: str = None) -> dict:
    """
    The actual enforcement entry point. Checks and redacts PII at BOTH
    boundaries -- not just the incoming question, but the outgoing
    answer too, since PII can leak through generation even when the
    input was clean (the model echoing back something from the question,
    or -- once real user documents are supported -- a retrieved chunk
    containing a real person's name/ID).

    Call this TWICE in practice:
      1. Before embedding/retrieval, with only user_question -- use
         result["redacted_question"] for everything downstream.
      2. After generation, with only generated_answer (leave
         user_question as None or reuse it) -- use
         result["redacted_answer"] as what's actually returned/stored.

    Or call it once with both, if you want a single combined check.
    """
    question_findings = check_for_pii(user_question) if user_question else []
    redacted_question = redact_pii(user_question) if user_question else None

    answer_findings = check_for_pii(generated_answer) if generated_answer else []
    redacted_answer = redact_pii(generated_answer) if generated_answer else None

    return {
        "original_question": user_question,
        "redacted_question": redacted_question,
        "pii_detected_in_question": len(question_findings) > 0,
        "question_findings": question_findings,

        "original_answer": generated_answer,
        "redacted_answer": redacted_answer,
        "pii_detected_in_answer": len(answer_findings) > 0,
        "answer_findings": answer_findings,
    }


if __name__ == "__main__":
    # Quick manual test -- PII in both the question AND a (contrived)
    # answer that echoes it back, to confirm both sides get redacted
    test_question = "My name is John Smith and my SSN is 234-56-7891, is my knee surgery covered?"
    test_answer = "Yes John Smith, your knee surgery is covered at 15% coinsurance after deductible. (Source: Anthem_EOC.pdf, Page 24)"

    result = pii_guardrail(test_question, test_answer)
    import json
    print(json.dumps(result, indent=2))