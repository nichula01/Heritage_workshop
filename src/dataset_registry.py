from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class DatasetInfo:
    name: str
    task: str
    modality: str
    annotation_type: str
    root_hint: str
    notes: str


DATASET_REGISTRY: Dict[str, DatasetInfo] = {
    "artpedia": DatasetInfo(
        name="ArtPedia",
        task="vision-language / artwork understanding",
        modality="image-text",
        annotation_type="captions / visual attributes / artistic metadata",
        root_hint="data/raw/artpedia",
        notes="Potentially useful for VLM-style heritage or artwork understanding experiments."
    ),
    "iconclass": DatasetInfo(
        name="Iconclass",
        task="art image classification / semantic concepts",
        modality="image-label",
        annotation_type="concept labels",
        root_hint="data/raw/iconclass",
        notes="Can support retrieval, classification, or semantic grounding tasks."
    ),
    "artemis": DatasetInfo(
        name="ARTEmis",
        task="art emotion / explanation / caption-style understanding",
        modality="image-text",
        annotation_type="emotions / explanations / language descriptions",
        root_hint="data/raw/artemis",
        notes="Useful if we study VLM reasoning or language alignment on art/heritage-like imagery."
    ),
}


def list_datasets() -> List[str]:
    return sorted(DATASET_REGISTRY.keys())


def get_dataset_info(name: str) -> DatasetInfo:
    key = name.lower()
    if key not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {name}. Available: {list_datasets()}")
    return DATASET_REGISTRY[key]


if __name__ == "__main__":
    for dataset_name in list_datasets():
        info = get_dataset_info(dataset_name)
        print(f"{info.name}: {info.task} | {info.modality} | root={info.root_hint}")
