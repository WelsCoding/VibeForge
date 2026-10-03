# https://github.com/Comfy-Org/ComfyUI/blob/v0.36.0/comfy/text_encoders/krea2.py

import math
import numbers
import re

import torch

from backend import memory_management
from backend.args import dynamic_args
from backend.krea2 import KREA2_HAS_ATTENTION_BIAS, KREA2_TOKEN_WEIGHTS
from backend.text_processing import emphasis

from ._comfy import EMBEDDINGS, INF, TOKEN_WEIGHTS, SDClipModel, SDTokenizer


# Matches the default global strength of Krea2PromptWeight in kijai/ComfyUI-KJNodes.
KREA2_PROMPT_WEIGHT_STRENGTH = 1.0

_KREA2_WEIGHT_PATTERN = re.compile(r"\(([^():]+):\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)")
_QWEN_IM_START, _QWEN_USER, _QWEN_NL, _QWEN_IM_END = 151644, 872, 198, 151645
_QWEN_IMAGE_PAD = 151655


def parse_krea2_prompt_weights(text: str) -> tuple[str, list[tuple[str, float]]]:
    """Return prompt text without Krea weight markup and its weighted phrases."""
    terms = [(match.group(1).strip(), float(match.group(2))) for match in _KREA2_WEIGHT_PATTERN.finditer(text)]
    clean_text = _KREA2_WEIGHT_PATTERN.sub(lambda match: match.group(1), text)
    return clean_text, terms


def _user_content_span(token_ids: list[int]) -> tuple[int | None, int | None]:
    for index in range(len(token_ids) - 2):
        if token_ids[index : index + 3] == [_QWEN_IM_START, _QWEN_USER, _QWEN_NL]:
            start = index + 3
            try:
                end = token_ids.index(_QWEN_IM_END, start)
            except ValueError:
                end = len(token_ids)
            return start, end
    return None, None


def _find_subsequence(sequence: list[int], subsequence: list[int], start: int, end: int) -> list[int]:
    if not subsequence:
        return []
    return [index for index in range(start, end - len(subsequence) + 1) if sequence[index : index + len(subsequence)] == subsequence]


def _qwen3vl_image_token_count(image: torch.Tensor) -> int:
    """Return the number of Qwen3-VL image embeddings inserted for a Krea reference."""
    if image.ndim == 4:
        height, width = image.shape[1:3]
    else:
        height, width = image.shape[:2]

    # Keep this in sync with process_qwen2vl_images(): it rounds images to 32-pixel
    # grids, then the Qwen3-VL merger combines each 2x2 group of 16px patches.
    min_pixels, max_pixels, factor = 3136, 12845056, 32
    height_bar = round(height / factor) * factor
    width_bar = round(width / factor) * factor

    if height_bar * width_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        height_bar = max(factor, math.floor(height / beta / factor) * factor)
        width_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif height_bar * width_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        height_bar = math.ceil(height * beta / factor) * factor
        width_bar = math.ceil(width * beta / factor) * factor

    grid_height = height_bar // 16
    grid_width = width_bar // 16
    return (grid_height * grid_width) // 4


class Qwen3VL4BEngine:
    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = SDClipModel(text_encoder, layer=[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35], layer_idx=None, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, enable_attention_masks=True, return_attention_masks=True)
        self.tokenizer = SDTokenizer(tokenizer, pad_with_end=False, has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=INF, min_length=1, pad_token=151643)

        self.llama_template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

        self.vision_block = "<|vision_start|><|image_pad|><|vision_end|>"

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.EmphasisNone()

    def tokenize(self, texts: str | list[str]) -> EMBEDDINGS | list[EMBEDDINGS]:
        if isinstance(texts, str):
            clean_text, _ = parse_krea2_prompt_weights(texts)
            return self.tokenizer.tokenizer(self.llama_template.format(clean_text))["input_ids"]
        else:
            clean_texts = [parse_krea2_prompt_weights(text)[0] for text in texts]
            return [self.tokenizer.tokenizer(self.llama_template.format(text))["input_ids"] for text in clean_texts]

    def __call__(self, texts: list[str], images: list[torch.Tensor] | None = None, apply_prompt_weights: bool = True) -> list[torch.Tensor] | dict[str, list[torch.Tensor]]:
        if any(emphasis.uses_emphasis(text) for text in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = "None"

        images = images or []
        parsed_texts = []
        for line in texts:
            if apply_prompt_weights:
                clean_text, terms = parse_krea2_prompt_weights(line)
            else:
                clean_text, terms = line, []
            parsed_texts.append((clean_text, terms))

        has_weight_markup = apply_prompt_weights and any(weight != 1.0 for _, terms in parsed_texts for _, weight in terms)

        zs: list[torch.Tensor] = []
        token_weight_rows: list[torch.Tensor] = []
        attention_bias_rows: list[torch.Tensor] = []
        cache: dict[str, torch.Tensor] = {}

        for clean_text, terms in parsed_texts:
            if clean_text in cache:
                cond = cache[clean_text]
            else:
                chunk = self._tokenize_with_weights(clean_text, images)
                cond = self.text_encoder.encode_token_weights(chunk)[0]
                token_pairs = chunk[0]

                token_ids = [int(item[0]) if not torch.is_tensor(item[0]) and isinstance(item[0], numbers.Integral) else -1 for item in token_pairs]
                template_start, _ = _user_content_span(token_ids)
                if template_start is None:
                    template_start = 0

                cond = cond[:, :, template_start:]
                cond = cond.permute(0, 2, 1, 3).reshape(-1, 12, 2560)
                cache[clean_text] = cond

            zs.append(cond)

            if has_weight_markup:
                weights, has_attention_bias = self._get_token_weights(clean_text, terms, cond.shape[0], images)
                token_weight_rows.append(weights)
                attention_bias_rows.append(torch.tensor([has_attention_bias], dtype=torch.bool))

        if not has_weight_markup:
            return zs

        return {
            "crossattn": zs,
            KREA2_TOKEN_WEIGHTS: token_weight_rows,
            KREA2_HAS_ATTENTION_BIAS: attention_bias_rows,
        }

    def _tokenize_ids(self, text: str) -> list[int]:
        token_ids = self.tokenizer.tokenizer(text)["input_ids"]
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return [int(token_id) for token_id in token_ids]

    def _get_token_weights(self, text: str, terms: list[tuple[str, float]], sequence_length: int, images: list[torch.Tensor]) -> tuple[torch.Tensor, bool]:
        weights = torch.ones((sequence_length, 2), dtype=torch.float32)
        weights[:, 1] = 0.0

        full_text = self.llama_template.format(self.vision_block * len(images) + text.strip())
        token_ids = self._tokenize_ids(full_text)
        user_start, user_end = _user_content_span(token_ids)
        if user_start is None:
            return weights, False

        image_positions = [index for index in range(user_start, user_end) if token_ids[index] == _QWEN_IMAGE_PAD]
        image_lengths = [_qwen3vl_image_token_count(image) for image in images[: len(image_positions)]]
        image_offsets = dict(zip(image_positions, image_lengths))

        has_attention_bias = False
        for phrase, weight in terms:
            if weight == 1.0:
                continue

            matches = []
            phrase_tokens = []
            for variant in (" " + phrase, phrase):
                variant_text = self.llama_template.format(variant)
                variant_ids = self._tokenize_ids(variant_text)
                phrase_start, phrase_end = _user_content_span(variant_ids)
                if phrase_start is None:
                    continue

                phrase_tokens = variant_ids[phrase_start:phrase_end]
                matches = _find_subsequence(token_ids, phrase_tokens, user_start, user_end)
                if matches:
                    break

            if not matches:
                memory_management.logger.warning(f"[Krea2] Prompt-weight phrase was not found in the tokenized prompt: {phrase!r}")
                continue

            if weight > 1.0:
                value_factor = 1.0
                key_bias = KREA2_PROMPT_WEIGHT_STRENGTH * (weight - 1.0) * 2.0
            else:
                value_factor = 1.0 + KREA2_PROMPT_WEIGHT_STRENGTH * (weight - 1.0)
                key_bias = 0.0

            for match_start in matches:
                for token_position in range(match_start, match_start + len(phrase_tokens)):
                    condition_position = token_position - user_start
                    condition_position += sum(image_length - 1 for image_position, image_length in image_offsets.items() if image_position < token_position)
                    if not 0 <= condition_position < sequence_length:
                        continue

                    weights[condition_position, 0] *= value_factor
                    if key_bias != 0.0:
                        weights[condition_position, 1] = key_bias
                        has_attention_bias = True

        return weights, has_attention_bias

    def _tokenize_with_weights(self, text: str, images: list[torch.Tensor]) -> TOKEN_WEIGHTS:
        llama_text = self.llama_template.format(self.vision_block * len(images) + text.strip())
        tokens = self.tokenizer.tokenize_with_weights(llama_text, disable_weights=True)

        embed_count = 0

        for row in tokens:
            for index in range(len(row)):
                if isinstance(row[index][0], (int, float)) and row[index][0] == _QWEN_IMAGE_PAD:
                    if len(images) > embed_count:
                        row[index] = ({"type": "image", "data": images[embed_count], "original_type": "image"},) + row[index][1:]
                        embed_count += 1

        return tokens
