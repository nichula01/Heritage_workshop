import re
from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalItem:
    sentence: str
    score: float
    index: int


def split_into_sentences(text: str) -> List[str]:
    text = str(text or "").replace("\n", " ").strip()
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


class TfidfSentenceRetriever:
    def retrieve(self, question: str, description: str, top_k: int = 2) -> List[RetrievalItem]:
        sentences = split_into_sentences(description)
        if not sentences:
            return []

        corpus = [question] + sentences
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        X = vec.fit_transform(corpus)

        qv = X[0:1]
        sv = X[1:]
        sims = cosine_similarity(qv, sv).flatten()

        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievalItem(sentence=sentences[idx], score=float(score), index=int(idx))
            for idx, score in ranked
        ]
