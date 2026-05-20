from src.aer.prediction_normalizer import normalize_prediction


def test_normalize_prediction_strips_common_prefix():
    assert normalize_prediction("The caption says Example text.") == "Example text"


def test_normalize_prediction_maps_known_surface_forms():
    assert normalize_prediction("print") == "Printing"
    assert normalize_prediction("on the back") == "Back"
