from dataclasses import dataclass
from typing import List, Optional

from src.aer.router import HybridRouter
from src.aer.retriever import TfidfSentenceRetriever


@dataclass
class PreparedEvidence:
    sample_id: str
    template_id: str
    question: str
    route: str
    route_confidence: float
    route_source: str
    route_reason: str
    evidence_mode: str
    retrieved_sentences: List[str]
    retrieval_scores: List[float]


class AdaptiveEvidenceRouter:
    def __init__(self):
        self.router = HybridRouter()
        self.retriever = TfidfSentenceRetriever()

    def prepare(
        self,
        sample_id: str,
        template_id: str,
        question: str,
        description: str,
        top_k: int = 2
    ) -> PreparedEvidence:
        rr = self.router.predict(question=question, template_id=template_id)

        if rr.route == "visual":
            return PreparedEvidence(
                sample_id=sample_id,
                template_id=template_id,
                question=question,
                route=rr.route,
                route_confidence=rr.confidence,
                route_source=rr.source,
                route_reason=rr.reason,
                evidence_mode="image_only",
                retrieved_sentences=[],
                retrieval_scores=[]
            )

        retrieved = self.retriever.retrieve(question=question, description=description, top_k=top_k)
        selected = [x.sentence for x in retrieved]
        scores = [x.score for x in retrieved]

        evidence_mode = "text_only" if rr.route == "contextual" else "image_plus_text"

        return PreparedEvidence(
            sample_id=sample_id,
            template_id=template_id,
            question=question,
            route=rr.route,
            route_confidence=rr.confidence,
            route_source=rr.source,
            route_reason=rr.reason,
            evidence_mode=evidence_mode,
            retrieved_sentences=selected,
            retrieval_scores=scores
        )
