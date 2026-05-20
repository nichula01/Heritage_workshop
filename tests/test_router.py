import json

from src.aer.router import HybridRouter


def test_router_uses_template_map(tmp_path):
    route_map = tmp_path / "routes.json"
    route_map.write_text(json.dumps({"AUTHOR": "contextual"}), encoding="utf-8")

    router = HybridRouter(route_map)
    result = router.predict("What color is the object?", template_id="AUTHOR")

    assert result.route == "contextual"
    assert result.source == "template_map"


def test_router_keyword_fallback_for_visual_question(tmp_path):
    route_map = tmp_path / "routes.json"
    route_map.write_text("{}", encoding="utf-8")

    router = HybridRouter(route_map)
    result = router.predict("What color is the object?")

    assert result.route == "visual"
    assert result.source == "keyword_rule"
