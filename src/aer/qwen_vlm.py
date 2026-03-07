from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


class QwenVLM:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        else:
            dtype = torch.float32

        self.dtype = dtype

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="sdpa"
        )

        # keep image resolution moderate for stability and memory
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=1024 * 28 * 28,
            use_fast=False
        )

    def _generate_from_messages(self, messages, max_new_tokens: int = 32) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.device)

        # deterministic decoding to avoid multinomial sampling path
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()

    def answer_image_only(self, image_path: str, question: str, max_new_tokens: int = 32) -> str:
        image_path = Path(image_path).resolve()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {
                        "type": "text",
                        "text": (
                            "Answer the question about this cultural heritage image. "
                            "Give a short answer only.\n"
                            f"Question: {question}"
                        ),
                    },
                ],
            }
        ]
        return self._generate_from_messages(messages, max_new_tokens=max_new_tokens)

    def answer_image_plus_text(
        self,
        image_path: str,
        question: str,
        evidence_text: str,
        max_new_tokens: int = 48
    ) -> str:
        image_path = Path(image_path).resolve()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {
                        "type": "text",
                        "text": (
                            "Answer the question about this cultural heritage image using the provided evidence. "
                            "Give a short answer only.\n"
                            f"Question: {question}\n"
                            f"Evidence: {evidence_text}"
                        ),
                    },
                ],
            }
        ]
        return self._generate_from_messages(messages, max_new_tokens=max_new_tokens)

    def answer_text_only(self, question: str, evidence_text: str, max_new_tokens: int = 32) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Answer the question using only the provided evidence. "
                            "Give a short answer only.\n"
                            f"Question: {question}\n"
                            f"Evidence: {evidence_text}"
                        ),
                    },
                ],
            }
        ]
        return self._generate_from_messages(messages, max_new_tokens=max_new_tokens)
