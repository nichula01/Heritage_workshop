from src.aer.retriever import TfidfSentenceRetriever, split_into_sentences


def test_split_into_sentences_handles_empty_text():
    assert split_into_sentences("") == []


def test_retriever_returns_most_relevant_sentence():
    retriever = TfidfSentenceRetriever()
    description = (
        "The cultural asset is kept in a museum. "
        "The author of the cultural asset is Maria Rossi. "
        "The object is made from bronze."
    )

    results = retriever.retrieve("Who is the author of the cultural asset?", description, top_k=1)

    assert len(results) == 1
    assert "Maria Rossi" in results[0].sentence
    assert results[0].score >= 0
