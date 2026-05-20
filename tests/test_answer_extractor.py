from src.aer.answer_extractor import extract_answer


def test_extract_author_from_structured_evidence():
    evidence = "The author of the cultural asset is Maria Rossi. The object is dated to 1900."

    answer = extract_answer(
        question="Who is the author of the cultural asset?",
        evidence_text=evidence,
        template_id="AUTHOR",
    )

    assert answer == "Maria Rossi"


def test_extract_affixed_technique_printing():
    evidence = "The technical characteristics of the element attached to the cultural asset are made in print."

    answer = extract_answer(
        question="What are the technical characteristics of the element?",
        evidence_text=evidence,
        template_id="AFFIXEDTECHNIQUE",
    )

    assert answer == "Printing"
