import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ROUTE_MAP_PATH = Path("metadata/viscounth_route_map.json")


@dataclass
class RouteResult:
    route: str
    confidence: float
    source: str
    reason: str


class HybridRouter:
    def __init__(self, route_map_path: Path = ROUTE_MAP_PATH):
        with open(route_map_path, "r", encoding="utf-8") as f:
            self.template_map = json.load(f)

        self.visual_keywords = [
            "what color", "how many", "where is", "where in", "shape",
            "black and white", "what is shown", "what appears"
        ]
        self.contextual_keywords = [
            "author", "who made", "who is", "period", "date", "century",
            "material", "technique", "owner", "location", "where is it kept",
            "purpose", "function", "historical", "criterion"
        ]
        self.mixed_keywords = [
            "attached", "written", "inscription", "captions",
            "visible", "depicted", "represented", "subject"
        ]

    def predict(self, question: str, template_id: Optional[str] = None) -> RouteResult:
        q = " ".join(str(question).lower().strip().split())

        if template_id:
            template_id = str(template_id).strip()
            if template_id in self.template_map:
                return RouteResult(
                    route=self.template_map[template_id],
                    confidence=0.95,
                    source="template_map",
                    reason=f"template_id={template_id}"
                )

        visual_hits = [k for k in self.visual_keywords if k in q]
        contextual_hits = [k for k in self.contextual_keywords if k in q]
        mixed_hits = [k for k in self.mixed_keywords if k in q]

        if mixed_hits:
            return RouteResult("mixed", 0.75, "keyword_rule", f"mixed hits: {mixed_hits}")
        if visual_hits and contextual_hits:
            return RouteResult("mixed", 0.70, "keyword_rule", f"visual={visual_hits}, contextual={contextual_hits}")
        if contextual_hits:
            return RouteResult("contextual", 0.70, "keyword_rule", f"contextual hits: {contextual_hits}")
        if visual_hits:
            return RouteResult("visual", 0.70, "keyword_rule", f"visual hits: {visual_hits}")

        return RouteResult("mixed", 0.50, "fallback", "default mixed fallback")
