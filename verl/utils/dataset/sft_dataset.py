# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

"""
VRAG/VRAG에서
python -m verl.utils.dataset.sft_dataset   --input ./lsm_tmp/results/sft_dataset/train_10.parquet   --tokenizer Qwen/Qwen2.5-7B-Instruct
위 명령어 실행
"""

from typing import List, Union

import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer
import json

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from PIL import Image

class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer,
                 prompt_key='prompt',
                 prompt_dict_keys=None,
                 response_key='response',
                 response_dict_keys=None,
                 max_length=2048,
                 truncation='error'):
        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        self.prompt_dict_keys = [] if not prompt_dict_keys else prompt_dict_keys
        self.response_dict_keys = [] if not response_dict_keys else response_dict_keys

        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True)

    """def _read_files_and_tokenize(self):

        def series_to_item(ls):
            import pandas, numpy
            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.prompts = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            # type(x): pandas.core.series.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)
            except Exception:
                print(f'self.prompts={self.prompts}')
                raise
        self.prompts = self.prompts.tolist()
        self.responses = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            try:
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)
            except Exception:
                print(f'self.responses={self.responses}')
                raise
        self.responses = self.responses.tolist()"""
    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # parquet 파일 읽기
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)

        # 여러 개 parquet이면 concat
        self.dataframe = pd.concat(dataframes, ignore_index=True)

        # messages 컬럼 확인
        if "messages" not in self.dataframe.columns:
            raise ValueError(
                f"parquet 파일에 'messages' 컬럼이 없습니다. "
                f"현재 컬럼들: {list(self.dataframe.columns)}"
            )

        # Optional: initialize AutoProcessor for multimodal SFT (if available).
        self._processor = None
        self._image_token = None
        self._merge_length = None
        self._image_token_id = None
        # Cache chat-template marker token ids for fast/robust loss masking.
        self._im_start_id = None
        self._im_end_id = None
        self._assistant_prefix_ids: list[int] = []
        try:
            from transformers import AutoProcessor

            name_or_path = getattr(self.tokenizer, "name_or_path", None)
            if isinstance(name_or_path, str) and name_or_path:
                self._processor = AutoProcessor.from_pretrained(name_or_path, trust_remote_code=True)
                self._image_token = getattr(self._processor, "image_token", None)
                image_processor = getattr(self._processor, "image_processor", None)
                merge_size = getattr(image_processor, "merge_size", None) if image_processor is not None else None
                try:
                    merge_size = int(merge_size) if merge_size is not None else None
                except Exception:
                    merge_size = None
                if merge_size:
                    self._merge_length = merge_size ** 2
                if self._image_token:
                    try:
                        ids = self.tokenizer(self._image_token, add_special_tokens=False)["input_ids"]
                        if isinstance(ids, list) and ids and isinstance(ids[0], list):
                            ids = ids[0]
                        if isinstance(ids, list) and len(ids) == 1:
                            self._image_token_id = int(ids[0])
                    except Exception:
                        self._image_token_id = None
        except Exception:
            self._processor = None
            self._image_token = None
            self._merge_length = None
            self._image_token_id = None

        try:
            self._im_start_id = int(self.tokenizer.convert_tokens_to_ids("<|im_start|>"))
            self._im_end_id = int(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        except Exception:
            self._im_start_id = None
            self._im_end_id = None

        # Build the assistant message prefix token pattern:
        #   <|im_start|>assistant\n
        try:
            role_ids = self.tokenizer("assistant", add_special_tokens=False)["input_ids"]
            nl_ids = self.tokenizer("\n", add_special_tokens=False)["input_ids"]
            if isinstance(role_ids, list) and isinstance(nl_ids, list):
                self._assistant_prefix_ids = [int(x) for x in role_ids + nl_ids]
        except Exception:
            self._assistant_prefix_ids = []

    def __len__(self):
        #return len(self.prompts)
        return len(self.dataframe)

    """def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        # apply chat template
        prompt_chat = [{'role': 'user', 'content': prompt}]

        # string
        prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        response_chat_str = response + tokenizer.eos_token

        # tokenize
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors='pt', add_special_tokens=False)
        prompt_ids = prompt_ids_output['input_ids'][0]
        prompt_attention_mask = prompt_ids_output['attention_mask'][0]

        response_ids_output = tokenizer(response_chat_str, return_tensors='pt', add_special_tokens=False)
        response_ids = response_ids_output['input_ids'][0]
        response_attention_mask = response_ids_output['attention_mask'][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                          dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == 'left':
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
            elif self.truncation == 'right':
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            elif self.truncation == 'error':
                raise NotImplementedError(f'{sequence_length=} is larger than {self.max_length=}')
            else:
                raise NotImplementedError(f'Unknown truncation method {self.truncation}')

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[:min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask
        }"""
    def __getitem__(self, item):
        tokenizer = self.tokenizer
        messages = self.dataframe.iloc[item]["messages"]

        # 문자열(JSON)일 경우 dict 리스트로 변환
        if isinstance(messages, str):
            messages = json.loads(messages)

        # Convert messages to a chat template-compatible format, while collecting images.
        # - If AutoProcessor is available, we will also return pixel_values + image_grid_thw.
        base_messages, base_image_paths = _convert_train1_messages(messages)

        def _build_inputs(messages_in: list[Dict[str, Any]], image_paths_in: list[str]):
            images_local: list[Image.Image] = []
            image_inputs_local: Dict[str, Any] = {}
            converted_local: list[Dict[str, Any]] = list(messages_in)
            if self._processor is not None and image_paths_in:
                # Keep the preprocessing consistent with RL's process_image defaults (scaled by pixels).
                images_local = [_load_and_process_image(p) for p in image_paths_in]
                image_inputs_local = self._processor.image_processor(images_local, return_tensors="pt")

                # Build vision token placeholders similar to RL runtime.
                if self._image_token and self._merge_length and "image_grid_thw" in image_inputs_local:
                    vision_strs: list[str] = []
                    for grid in image_inputs_local["image_grid_thw"]:
                        try:
                            n = int(grid.prod().item()) // int(self._merge_length)
                        except Exception:
                            n = 1
                        n = max(1, n)
                        vision_strs.append(f"<|vision_start|>{self._image_token * n}<|vision_end|>")
                    converted_local = _inject_vision_strings(converted_local, vision_strs)
                else:
                    # If we can't inject vision tokens reliably, fall back to text-only for safety.
                    image_inputs_local = {}

            # Ensure chat template sees only string content. Any remaining image placeholders are blanked.
            converted_local_str: list[Dict[str, Any]] = []
            for m in converted_local:
                role = m.get("role")
                content = m.get("content")
                if role == "user" and isinstance(content, dict) and "_image_placeholders" in content:
                    converted_local_str.append({**m, "content": ""})
                else:
                    converted_local_str.append(m)

            chat_str_local = tokenizer.apply_chat_template(converted_local_str, tokenize=False)
            tokenized_local = tokenizer(chat_str_local, return_tensors="pt", add_special_tokens=False)
            input_ids_local = tokenized_local["input_ids"][0]
            attention_mask_local = tokenized_local["attention_mask"][0]

            # Guard: if images are present but no (or mismatched) image tokens are found, rebuild as text-only.
            if image_inputs_local and self._merge_length and "image_grid_thw" in image_inputs_local:
                if self._image_token_id is None:
                    return _build_inputs(messages_in, [])

                n_image_tokens = int((input_ids_local == int(self._image_token_id)).sum().item())
                try:
                    expected_tokens = 0
                    for grid in image_inputs_local["image_grid_thw"]:
                        expected_tokens += int(grid.prod().item()) // int(self._merge_length)
                except Exception:
                    expected_tokens = None

                if n_image_tokens == 0 or (expected_tokens is not None and n_image_tokens != expected_tokens):
                    return _build_inputs(messages_in, [])

            return converted_local_str, image_inputs_local, input_ids_local, attention_mask_local

        # Build initial (untruncated) inputs
        converted_messages, image_inputs, input_ids, attention_mask = _build_inputs(
            base_messages, base_image_paths
        )

        # Avoid truncating inside vision token blocks: if the sample is too long and includes images,
        # drop the earliest messages (and corresponding images) until it fits.
        if len(input_ids) > self.max_length and base_image_paths:
            if self.truncation == "error":
                raise RuntimeError(f"sequence_length={len(input_ids)} > max_length={self.max_length} for multimodal sample")

            messages_work = list(base_messages)
            image_paths_work = list(base_image_paths)
            while len(input_ids) > self.max_length and messages_work:
                m0 = messages_work.pop(0)
                c0 = m0.get("content")
                if m0.get("role") == "user" and isinstance(c0, dict) and "_image_placeholders" in c0:
                    try:
                        n_drop = int(c0.get("_image_placeholders") or 0)
                    except Exception:
                        n_drop = 0
                    if n_drop > 0:
                        image_paths_work = image_paths_work[n_drop:]

                converted_messages, image_inputs, input_ids, attention_mask = _build_inputs(
                    messages_work, image_paths_work
                )

            # If the multimodal sample is still too long (e.g., extreme vision token count),
            # fall back to a text-only version to avoid image-token/feature mismatches.
            if len(input_ids) > self.max_length and image_paths_work:
                converted_messages, image_inputs, input_ids, attention_mask = _build_inputs(
                    messages_work, []
                )

        # padding / truncation (text-only, or multimodal after message-level trimming)
        if len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])
        elif len(input_ids) > self.max_length:
            if self.truncation == "right":
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            elif self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
            else:
                raise NotImplementedError

        position_ids = compute_position_id_with_mask(attention_mask)

        # === loss_mask ===
        # Build loss_mask on *input positions* i where the *label token* (i+1) belongs to an assistant message.
        # This avoids brittle substring search and prevents mis-masking when content repeats.
        loss_mask = torch.zeros_like(attention_mask)
        if self._im_start_id is not None and self._im_end_id is not None and self._assistant_prefix_ids:
            seq_end = int(attention_mask.sum().item())
            ids = input_ids[:seq_end]
            assistant_token_mask = torch.zeros_like(ids, dtype=torch.long)

            prefix = [self._im_start_id] + self._assistant_prefix_ids
            prefix_len = len(prefix)
            prefix_t = torch.tensor(prefix, dtype=ids.dtype)

            i = 0
            while i <= len(ids) - prefix_len:
                if torch.equal(ids[i:i + prefix_len], prefix_t):
                    content_start = i + prefix_len
                    j = content_start
                    while j < len(ids) and int(ids[j].item()) != int(self._im_end_id):
                        j += 1
                    if j < len(ids) and int(ids[j].item()) == int(self._im_end_id):
                        # Mark assistant content tokens, including <|im_end|> (EOS).
                        assistant_token_mask[content_start:j + 1] = 1
                        i = j + 1
                        continue
                i += 1

            # Shift so loss_mask aligns with next-token loss positions.
            if len(assistant_token_mask) > 1:
                loss_mask[:seq_end - 1] = assistant_token_mask[1:]
            loss_mask = loss_mask * attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "sample_idx": torch.tensor(item, dtype=torch.long),
            # Optional multimodal fields (trainer may ignore if model doesn't support).
            "pixel_values": image_inputs.get("pixel_values") if image_inputs else None,
            "image_grid_thw": image_inputs.get("image_grid_thw") if image_inputs else None,
        }


@dataclass(frozen=True)
class _ConvertedMessages:
    messages: list[Dict[str, Any]]
    image_paths: list[str]


def _convert_train1_messages(messages: Any) -> Tuple[list[Dict[str, Any]], list[str]]:
    """
    Convert train1 schema messages into a tokenizer.apply_chat_template-friendly list,
    while collecting image paths from user messages of the form:
      {"role":"user","content":[{"image":"./path.jpg"}]}
    """
    if not isinstance(messages, list):
        return [], []
    converted: list[Dict[str, Any]] = []
    image_paths: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role == "user" and isinstance(content, list):
            # Keep a placeholder for now; actual vision tokens will be injected later.
            # Collect image paths in order. Multiple images in one message are supported.
            n = 0
            for it in content:
                if isinstance(it, dict) and isinstance(it.get("image"), str) and it["image"]:
                    image_paths.append(it["image"])
                    n += 1
            converted.append({"role": "user", "content": {"_image_placeholders": n}})
            continue

        if isinstance(content, list):
            # Fallback: stringify unsupported list content to keep the sample usable.
            content = json.dumps(content, ensure_ascii=False)
        converted.append({"role": role, "content": content})
    return converted, image_paths


def _inject_vision_strings(messages: list[Dict[str, Any]], vision_strs: list[str]) -> list[Dict[str, Any]]:
    """
    Replace each empty user message (created from an image-list user content) with the next
    vision token string. If counts mismatch, leftover placeholders stay empty.
    """
    out: list[Dict[str, Any]] = []
    j = 0
    for m in messages:
        content = m.get("content")
        if m.get("role") == "user" and isinstance(content, dict) and "_image_placeholders" in content:
            n = content.get("_image_placeholders")
            try:
                n = int(n)
            except Exception:
                n = 0
            parts: list[str] = []
            for _ in range(max(0, n)):
                if j < len(vision_strs):
                    parts.append(vision_strs[j])
                    j += 1
            out.append({**m, "content": "".join(parts)})
        else:
            out.append(m)
    return out


def _load_and_process_image(path: str) -> Image.Image:
    # Match vrag_agent.generation.process_image behavior (pixel-count-based resizing).
    import math

    img = Image.open(path)
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")

    max_pixels = 512 * 28 * 28
    min_pixels = 256 * 28 * 28
    pixels = img.width * img.height
    if pixels > max_pixels:
        resize_factor = math.sqrt(max_pixels / pixels)
        img = img.resize((int(img.width * resize_factor), int(img.height * resize_factor)))
    elif pixels < min_pixels:
        resize_factor = math.sqrt(min_pixels / pixels)
        img = img.resize((int(img.width * resize_factor), int(img.height * resize_factor)))
    return img


def sft_multimodal_collate_fn(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate for SFTDataset with optional multimodal fields.
    Pads pixel_values/image_grid_thw along the num_images dimension.
    """
    out: Dict[str, Any] = {}
    # Stack fixed-size tensors
    for k in ("input_ids", "attention_mask", "position_ids", "loss_mask", "sample_idx"):
        out[k] = torch.stack([b[k] for b in batch], dim=0)

    # Optional multimodal (Qwen2.5-VL / transformers 4.51):
    # - Per-sample processor output:
    #   - pixel_values: (n_patches_total, hidden_dim)
    #   - image_grid_thw: (n_images, 3)
    #
    # We must keep the leading batch dimension to satisfy TensorDict validation and
    # micro-batch splitting. We therefore pad to:
    #   - pixel_values: (B, max_patches, hidden_dim) + patch_attention_mask (B, max_patches)
    #   - image_grid_thw: (B, max_images, 3) + image_attention_mask (B, max_images)
    #
    # The trainer will flatten valid patches/images right before model.forward to match
    # Qwen2.5-VL expected shapes: (total_patches, hidden_dim) and (total_images, 3).
    pixel_values_list = [b.get("pixel_values") for b in batch]
    grid_list = [b.get("image_grid_thw") for b in batch]

    ref_p = next((p for p in pixel_values_list if isinstance(p, torch.Tensor) and p.numel() > 0), None)
    if ref_p is None:
        out["pixel_values"] = None
        out["patch_attention_mask"] = None
        out["image_grid_thw"] = None
        out["image_attention_mask"] = None
        return out

    hidden_dim = int(ref_p.shape[-1])
    dtype_p = ref_p.dtype

    patch_counts = [int(p.shape[0]) if isinstance(p, torch.Tensor) else 0 for p in pixel_values_list]
    max_patches = max(patch_counts) if patch_counts else 0
    image_counts = [int(g.shape[0]) if isinstance(g, torch.Tensor) else 0 for g in grid_list]
    max_images = max(image_counts) if image_counts else 0

    pixel_values = torch.zeros((len(batch), max_patches, hidden_dim), dtype=dtype_p)
    patch_attention_mask = torch.zeros((len(batch), max_patches), dtype=torch.long)
    image_grid_thw = torch.zeros((len(batch), max_images, 3), dtype=torch.long)
    image_attention_mask = torch.zeros((len(batch), max_images), dtype=torch.long)

    for i, (p, g) in enumerate(zip(pixel_values_list, grid_list)):
        if isinstance(p, torch.Tensor) and p.numel() > 0:
            n = int(p.shape[0])
            pixel_values[i, :n] = p
            patch_attention_mask[i, :n] = 1
        if isinstance(g, torch.Tensor) and g.numel() > 0:
            g = g.to(torch.long)
            m = int(g.shape[0])
            image_grid_thw[i, :m] = g
            image_attention_mask[i, :m] = 1

    out["pixel_values"] = pixel_values
    out["patch_attention_mask"] = patch_attention_mask
    out["image_grid_thw"] = image_grid_thw
    out["image_attention_mask"] = image_attention_mask
    return out

    """def __getitem__(self, item):
        tok = self.tokenizer
        row = self.dataframe.iloc[item]
        messages = row["messages"]

        # parquet에 문자열(JSON)로 저장된 경우 파싱
        if isinstance(messages, str):
            messages = json.loads(messages)

        # content가 list면 JSON 문자열로 변환 (chat template는 문자열을 기대)
        for m in messages:
            if isinstance(m.get("content"), list):
                m["content"] = json.dumps(m["content"], ensure_ascii=False)

        # 전체 대화 문자열 (Qwen ChatML 가정)
        chat_str = tok.apply_chat_template(messages, tokenize=False)

        enc = tok(chat_str, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].squeeze(0)        # (L,)
        attention_mask = enc["attention_mask"].squeeze(0)

        # ---- pad / truncation ----
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        L = input_ids.numel()
        if L < self.max_length:
            pad_len = self.max_length - L
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=input_ids.dtype)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])
        elif L > self.max_length:
            if self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
            else:
                raise RuntimeError(f"sequence_length={L} > max_length={self.max_length}")

        position_ids = compute_position_id_with_mask(attention_mask)

        # ---- loss_mask: <|im_start|>assistant ... <|im_end|> 본문만 1
        loss_mask = torch.zeros_like(attention_mask)

        # 토큰 헬퍼
        def toks(s: str) -> torch.Tensor:
            return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)

        # 경계 토큰(개행 유무 모두 대응)
        start_assistant_a = toks("<|im_start|>assistant")
        start_assistant_b = toks("<|im_start|>assistant\n")
        end_tok = toks("<|im_end|>")

        # search_complete 태그 (우선 true 포함, 없으면 닫힘태그만이라도)
        sc_true = toks("<search_complete>true</search_complete>")
        sc_close = toks("</search_complete>")

        # 부분열 검색 (간단 선형)
        def find_subseq(hay: torch.Tensor, nee: torch.Tensor, st: int = 0) -> int:
            Lh, Ln = hay.numel(), nee.numel()
            if Ln == 0 or Lh < Ln:
                return -1
            for i in range(st, Lh - Ln + 1):
                if torch.equal(hay[i:i+Ln], nee):
                    return i
            return -1

        # 블록 찾기
        def find_blocks(ids: torch.Tensor, start_tok: torch.Tensor) -> list[tuple[int,int]]:
            blocks, cur = [], 0
            while True:
                s = find_subseq(ids, start_tok, cur)
                if s < 0:
                    break
                cs = s + start_tok.numel()           # 본문 시작
                e = find_subseq(ids, end_tok, cs)
                if e < 0:
                    break
                blocks.append((cs, e))               # [본문시작, end_tok 시작)
                cur = e + end_tok.numel()
            return blocks

        blocks = find_blocks(input_ids, start_assistant_b) + find_blocks(input_ids, start_assistant_a)
        blocks = sorted(blocks, key=lambda x: x[0])

        # 중복/겹침 제거(개행 유무로 두 번 잡힌 경우)
        merged = []
        for b in blocks:
            if not merged or b[0] >= merged[-1][1]:
                merged.append(b)
            # 겹치면 건너뜀

        # 각 블록에서 search_complete까지 마스킹
        for (cs, e) in merged:
            ce = min(e, input_ids.numel())
            rel = input_ids[cs:ce]

            pos = find_subseq(rel, sc_true, 0)
            if pos >= 0:
                cut_end = cs + pos + sc_true.numel()
            else:
                pos2 = find_subseq(rel, sc_close, 0)
                if pos2 >= 0:
                    cut_end = cs + pos2 + sc_close.numel()
                else:
                    cut_end = ce

            cut_end = min(cut_end, input_ids.numel())
            # 마지막 토큰(보통 EOS 성격)은 예측 제외
            if cut_end - cs > 1:
                loss_mask[cs:cut_end] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }"""





        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=2048)
    #parser.add_argument("--num_samples", type=int, default=2)
    args = parser.parse_args()

    # === 데이터셋 로드 ===
    dataset = SFTDataset(
        parquet_files=args.input,
        tokenizer=args.tokenizer,
        max_length=args.max_len,
        truncation="right"
    )
    print(f"✅ parquet 로드 완료: {len(dataset)} samples")

    results = []

    # === 샘플 점검 & JSON 생성 ===
    for i in range(len(dataset)):
        item = dataset[i]
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]
        tokenizer = dataset.tokenizer

        # loss_mask=1 인 위치
        assistant_ids = input_ids[loss_mask == 1]

        # 디코딩된 텍스트
        assistant_text = tokenizer.decode(assistant_ids)

        # 원래 parquet 안의 id 필드 사용
        # 만약 parquet에 'id'가 없다면 'uid' 등 다른 키 이름 확인 필요
        sample_id = item.get("uid", i)  # 'id'가 없으면 인덱스 사용

        results.append({
            "id": sample_id,
            "assistant_text": assistant_text
        })


    # === 원본 parquet 옆에 저장 경로 만들기 ===
    base, ext = os.path.splitext(args.input)
    output_file = base + "_assistant_only.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 저장 완료: {output_file}")
