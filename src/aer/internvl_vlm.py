from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []

    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def load_image(image_file: str, input_size: int = 448, max_num: int = 6):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVLVLM:
    def __init__(self, model_name: str = "OpenGVLab/InternVL2_5-2B"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )

    def answer_image_only(self, image_path: str, question: str, max_new_tokens: int = 32) -> str:
        pixel_values = load_image(image_path, max_num=6).to(self.dtype).to(self.device)
        prompt = (
            "<image>\n"
            "Answer using only the image.\n"
            "Return only the shortest answer phrase.\n"
            "Do not explain.\n"
            f"Question: {question}"
        )
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
        response = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config)
        return str(response).strip()

    def answer_image_plus_text(
        self,
        image_path: str,
        question: str,
        evidence_text: str,
        max_new_tokens: int = 48
    ) -> str:
        pixel_values = load_image(image_path, max_num=6).to(self.dtype).to(self.device)
        prompt = (
            "<image>\n"
            "Use both the image and the provided evidence.\n"
            "If the evidence explicitly states the answer, copy it exactly.\n"
            "Return only the shortest answer phrase.\n"
            "Do not explain.\n"
            f"Question: {question}\n"
            f"Evidence: {evidence_text}"
        )
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
        response = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config)
        return str(response).strip()

    def answer_text_only(
        self,
        question: str,
        evidence_text: str,
        max_new_tokens: int = 32
    ) -> str:
        prompt = (
            "Answer using only the provided evidence.\n"
            "Extract the shortest exact answer span if possible.\n"
            "Return only the answer phrase.\n"
            "Do not explain.\n"
            f"Question: {question}\n"
            f"Evidence: {evidence_text}"
        )
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
        response = self.model.chat(self.tokenizer, None, prompt, generation_config)
        return str(response).strip()
