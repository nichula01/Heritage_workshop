from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer


class MiniCPMVLM:
    def __init__(self, model_name: str = "openbmb/MiniCPM-V-2_6"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=dtype
        )
        self.model = self.model.eval().to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

    def answer_image_only(self, image_path: str, question: str) -> str:
        image = Image.open(Path(image_path)).convert("RGB")
        msgs = [{"role": "user", "content": [image, question]}]
        res = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer
        )
        return str(res).strip()

    def answer_image_plus_text(self, image_path: str, question: str, evidence_text: str) -> str:
        image = Image.open(Path(image_path)).convert("RGB")
        prompt = (
            "Answer the question about this cultural heritage image using the provided evidence. "
            "Give a short answer only.\n"
            f"Question: {question}\n"
            f"Evidence: {evidence_text}"
        )
        msgs = [{"role": "user", "content": [image, prompt]}]
        res = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer
        )
        return str(res).strip()
