import torch
import re
import numpy as np
from collections import defaultdict, deque
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
from transformers.image_processing_base import BatchFeature
from PIL import Image
from tqdm import tqdm
import json
#generator 수정
import uuid

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time as _time
import random as _random
import asyncio
import base64 

# ▼▼▼[성능 측정 추가]▼▼▼ 수정
# GPUMonitor와 시간 기록을 위한 모듈을 가져옵니다.
from lsm_tmp.gpu_monitor import GPUMonitor
from datetime import datetime
# ▲▲▲[성능 측정 추가]▲▲▲

# =============================================================================
# [Phase 2] 텐서 연산 최적화용 로깅 유틸리티
# =============================================================================
import logging

# Phase 2 전용 로거 설정
_phase2_logger = logging.getLogger("generation.phase2_optimization")
_phase2_logger.setLevel(logging.DEBUG)

# 콘솔 핸들러 (INFO 레벨 이상)
if not _phase2_logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setLevel(logging.INFO)
    _console_handler.setFormatter(logging.Formatter(
        '[Phase2] %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    ))
    _phase2_logger.addHandler(_console_handler)

# 성능 측정 환경변수 플래그 (기본값: False)
_PHASE2_PERF_LOG_ENABLED = os.getenv("PHASE2_PERF_LOG", "0") == "1"


class Phase2PerfTimer:
    """Phase 2 최적화 함수의 성능 측정을 위한 컨텍스트 매니저"""

    def __init__(self, func_name: str, batch_size: int = None):
        self.func_name = func_name
        self.batch_size = batch_size
        self.start_time = None
        self.elapsed_ms = None

    def __enter__(self):
        if _PHASE2_PERF_LOG_ENABLED:
            self.start_time = _time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if _PHASE2_PERF_LOG_ENABLED and self.start_time:
            self.elapsed_ms = (_time.perf_counter() - self.start_time) * 1000
            batch_info = f" (batch_size={self.batch_size})" if self.batch_size else ""
            _phase2_logger.info(f"{self.func_name}{batch_info}: {self.elapsed_ms:.3f}ms")
        return False


# ===== (1) DashScope 설정 =====
from http import HTTPStatus
from dotenv import load_dotenv

# dotenv_dir = '/home/isdslab/sangmin/VRAG_test/'  # 기존 하드코딩 경로
dotenv_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트

# 2. .env 파일의 전체 경로를 만듭니다.
dotenv_path = os.path.join(dotenv_dir, '.env')

# 3. 해당 경로의 .env 파일을 명시적으로 로드합니다.
load_dotenv(dotenv_path=dotenv_path)

try:
    import dashscope  # frozen generator (Qwen2.5-VL-72B 계열)
    import os as _os
    dashscope.base_http_api_url = _os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope-intl.aliyuncs.com/api/v1"
    )
    _API_KEY = _os.getenv("DASHSCOPE_API_KEY") or _os.getenv("DASH_SCOPE_KEY")
    if not _API_KEY:
        raise RuntimeError("Set DASHSCOPE_API_KEY (or DASH_SCOPE_KEY).")
    dashscope.api_key = _API_KEY
    _HAS_DASHSCOPE = True
except Exception:
    _HAS_DASHSCOPE = False

# >>> ADDED: DashScope 멀티모달 헬퍼 (import 블록 바로 아래에 추가)
try:
    from dashscope import MultiModalConversation
except Exception:
    pass  # _HAS_DASHSCOPE=False 인 경우 대비

def _extract_text_from_multimodal(resp):
    """DashScope 멀티모달 응답에서 텍스트를 최대한 안전하게 추출"""
    try:
        ot = getattr(resp, "output_text", None)
        if ot:
            return str(ot).strip()
    except Exception:
        pass

    out = getattr(resp, "output", None)
    if not isinstance(out, dict):
        return None

    choices = out.get("choices") or []
    if not choices:
        return None
    msg = choices[0].get("message") or {}
    content = msg.get("content") or []
    texts = []
    for part in content:
        if isinstance(part, dict) and part.get("text") is not None:
            texts.append(str(part["text"]))
    if texts:
        return "".join(texts).strip()

    if msg.get("text") is not None:
        return str(msg["text"]).strip()
    if out.get("text") is not None:
        return str(out["text"]).strip()
    return None


def _dashscope_call_with_fallback(model: str, messages: list, max_tokens: int):
    """SDK 버전 호환: max_output_tokens → 실패 시 max_tokens로 재시도"""
    try:
        return MultiModalConversation.call(
            model=model,
            messages=messages,
            max_output_tokens=max_tokens,
        )
    except TypeError:
        pass  # 일부 SDK는 max_output_tokens 미지원
    return MultiModalConversation.call(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )

def _to_image_part(path: str) -> dict | None:
    """로컬 경로를 DashScope 이미지 파트(dict)로 변환 (file:// 스킴 강제)"""
    if not path:
        return None
    if not path.startswith("file://"):
        path = "file://" + os.path.abspath(path)
    return {"image": path}
# <<< ADDED 끝


# =============================================================================
# [Phase 5] OpenAI SDK 비동기 클라이언트 (Frozen Generator 최적화)
# - DashScope OpenAI 호환 API 사용
# - AsyncOpenAI로 진정한 비동기 병렬 처리
# - 기존 동기 인터페이스 유지하면서 내부적으로 비동기 처리
# =============================================================================
_OPENAI_ASYNC_CLIENT = None
_HAS_OPENAI_ASYNC = False

try:
    from openai import AsyncOpenAI

    _DASHSCOPE_OPENAI_BASE_URL = os.getenv(
        "DASHSCOPE_OPENAI_BASE_URL",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    _DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASH_SCOPE_KEY")

    if _DASHSCOPE_API_KEY:
        _OPENAI_ASYNC_CLIENT = AsyncOpenAI(
            api_key=_DASHSCOPE_API_KEY,
            base_url=_DASHSCOPE_OPENAI_BASE_URL,
            timeout=60.0,
            max_retries=0,  # 우리가 직접 재시도 로직 관리
        )
        _HAS_OPENAI_ASYNC = True
        _phase2_logger.info(f"[Phase5] OpenAI AsyncClient initialized: {_DASHSCOPE_OPENAI_BASE_URL}")
except ImportError:
    _phase2_logger.warning("[Phase5] OpenAI SDK not installed. Falling back to DashScope SDK.")
except Exception as e:
    _phase2_logger.warning(f"[Phase5] Failed to initialize OpenAI AsyncClient: {e}")


def _image_to_base64_url(path: str) -> str | None:
    """이미지 파일을 base64 data URL로 변환 (OpenAI Vision API 호환)"""
    if not path or not os.path.exists(path):
        return None

    try:
        # 확장자로 MIME 타입 결정
        ext = os.path.splitext(path)[1].lower()
        mime_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
        }
        mime_type = mime_map.get(ext, 'image/jpeg')

        with open(path, 'rb') as f:
            image_data = f.read()

        base64_data = base64.b64encode(image_data).decode('utf-8')
        return f"data:{mime_type};base64,{base64_data}"
    except Exception:
        return None


async def _call_frozen_generator_async_single(
    client: 'AsyncOpenAI',
    model: str,
    question: str,
    image_paths: List[str],
    max_tokens: int = 1024,
    semaphore: asyncio.Semaphore = None,
) -> Tuple[int, str]:
    """
    OpenAI 호환 API를 사용한 비동기 단일 호출

    Args:
        client: AsyncOpenAI 클라이언트
        model: 모델명 (예: "qwen2.5-vl-72b-instruct")
        question: 질문 텍스트
        image_paths: 이미지 경로 리스트
        max_tokens: 최대 토큰 수
        semaphore: 동시 요청 수 제한용 세마포어

    Returns:
        (status_code, answer_text) 튜플
    """
    if semaphore:
        async with semaphore:
            return await _call_frozen_generator_async_single_impl(
                client, model, question, image_paths, max_tokens
            )
    else:
        return await _call_frozen_generator_async_single_impl(
            client, model, question, image_paths, max_tokens
        )


async def _call_frozen_generator_async_single_impl(
    client: 'AsyncOpenAI',
    model: str,
    question: str,
    image_paths: List[str],
    max_tokens: int = 1024,
) -> Tuple[int, str]:
    """비동기 호출 구현부"""
    try:
        qtext = (question or "").strip() or "."

        sys_prompt = (
            "You are a visual QA generator. "
            "Use only the provided images and the user question. "
            "Return ONLY the final answer text without extra explanations."
        )

        # OpenAI Vision API 형식으로 메시지 구성
        user_content = []

        # 이미지를 base64로 인코딩하여 추가
        for p in (image_paths or []):
            base64_url = _image_to_base64_url(p)
            if base64_url:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": base64_url}
                })

        # 텍스트 질문 추가
        user_content.append({
            "type": "text",
            "text": f"Question: {qtext}"
        })

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content}
        ]

        # 비동기 API 호출
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
        )

        # 응답 추출
        if response.choices and len(response.choices) > 0:
            answer = response.choices[0].message.content or ""
            return (200, answer.strip())

        return (200, "")

    except Exception as e:
        error_str = str(e).lower()
        # Rate limit 또는 서버 오류 감지
        if "rate" in error_str or "429" in error_str:
            return (429, "")
        elif "500" in error_str or "502" in error_str or "503" in error_str:
            return (503, "")
        elif "timeout" in error_str:
            return (408, "")
        else:
            return (0, "")


# =============================================================================
# [Phase 4] 이미지 로딩 캐싱 (LRU Cache)
# - 동일 이미지에 대한 반복 로딩 시 디스크 I/O 회피
# - maxsize=64: 배치 크기 고려한 캐시 크기 (메모리 vs 성능 트레이드오프)
# =============================================================================
from functools import lru_cache

@lru_cache(maxsize=64)
def _cached_image_open(path: str) -> 'Image.Image':
    """
    캐시된 이미지 로딩 함수

    동일 경로에 대한 반복 호출 시 캐시에서 반환합니다.
    주의: 반환된 이미지는 원본이므로 수정 시 .copy() 필요
    """
    return Image.open(path)


def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))
    elif isinstance(image, str):
        image = Image.open(image)


    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image

#수정 추가
FORCED_COMPLETION_RESPONSE = "<think>Maximum turn limit reached. Trigger search_complete.</think><search_complete>true</search_complete>"

# =============================================================================
# [Phase 1] 사전 컴파일된 정규식 패턴 (성능 최적화)
# - 모듈 로드 시 1회만 컴파일하여 재사용
# - 매 호출마다 re.compile() 오버헤드 제거
# =============================================================================
_RE_EXTRACT_TAGS = re.compile(r"<(search|think|bbox|search_complete)>(.*?)</\1>", re.DOTALL)
_RE_ACTION_PATTERN = re.compile(r'<(search|bbox|search_complete)>(.*?)</\1>', re.DOTALL)
_RE_UID_SUFFIX = re.compile(r'(\d+)$')

# =============================================================================
# [Phase 3] 루프 최적화용 상수
# - 루프 내에서 반복 생성되던 문자열을 모듈 레벨 상수로 추출
# - 가독성 향상 및 유지보수 용이
# =============================================================================
_MSG_INVALID_BBOX = (
    '\n<|im_start|>user\n'
    'Your previous action is invalid. \n'
    ' The bbox format is invalid. Expected format: JSON array [x1, y1, x2, y2] with all values >= 0. '
    'Please try again.\n'
    '<|im_end|>\n<|im_start|>assistant\n'
)

_MSG_INVALID_ACTION = (
    '\n<|im_start|>user\n'
    'Your previous action is invalid. '
    'You must conduct reasoning inside <think> and </think> every time you get new information. '
    'After reasoning, if you find you lack some knowledge, you can call a search engine using <search> query </search> and the user will return the search results. '
    'Whenever you retrieve an image, you may crop it for a clearer view using <bbox>[x1, y1, x2, y2]</bbox>. '
    'You can search as many times as you want. '
    'If you determine that no further external knowledge is needed, you must finish with <search_complete>true</search_compelte>. '
    'Otherwise, continue with <search> or <bbox> actions until you are ready to finish. '
    'Please try again.\n'
    '<|im_end|>\n<|im_start|>assistant\n'
)

# run_llm_loop에서 사용하는 키 리스트 (루프 불변)
_CUT_KEYS = ['input_ids', 'attention_mask', 'position_ids']

@dataclass
class GenerationConfig:
    max_turns: int
    max_prompt_length: int
    num_gpus: int
    search_url: str = None
    #generator added
    crops_dir: str = "./agent_crops"
    frozen_model: str = "qwen2.5-vl-72b-instruct"   # Qwen2.5-VL-72B-Instruct 호환
    frozen_max_tokens: int = 1024
    generator_max_images: int = 8
    use_system_prompt: bool = True
    generator_batch_workers: int = 4
    frozen_max_retries: int = 3
    frozen_backoff_base: float = 1.5
    # [Phase 5] OpenAI 비동기 API 설정
    frozen_max_concurrent: int = 50          # 동시 API 요청 수 (비동기 모드)
    # [NEW] 검색 최적화 옵션
    async_search: bool = True                # 비동기 병렬 검색 활성화
    search_batch_size: int = 512             # 검색 요청 배치 크기
    search_max_workers: int = 16              # 병렬 검색 워커 수 (4→8로 증가)
    search_timeout: int = 60                 # 검색 타임아웃 (초) - 5초→60초로 증가
    # [Phase 7] Tool 호출 완전 비동기화
    phase7_tool_async: bool = True           # Phase 7 비동기화 활성화 (기본: True)
    


class LLMGenerationManager:
    def __init__(
        self,
        processor,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
        streaming_reward_manager=None,  # [NEW] 스트리밍 Reward Manager
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=self.tokenizer.pad_token_id
        ))
        #generator added
        os.makedirs(self.config.crops_dir, exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
        self.cropped_images = None
        self.questions = None

        # [NEW] 스트리밍 Reward Manager
        self.streaming_reward_manager = streaming_reward_manager
        self._prompt_completion_status: Dict[str, Dict] = {}

        # [Phase 4] 비동기 이미지 저장을 위한 ThreadPoolExecutor
        # - 이미지 저장은 I/O 바운드 작업으로 GIL 영향 적음
        # - max_workers=4: 일반적인 디스크 I/O 병렬 처리에 적합
        self._save_executor = ThreadPoolExecutor(max_workers=4)
        self._pending_saves: List = []  # 완료 대기 중인 저장 작업

        # [최적화] HTTP 연결 풀링 - 연결 재사용으로 20-30% 성능 향상
        self._search_session = requests.Session()
        # 연결 풀 크기 설정 (워커 수에 맞춤)
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=self.config.search_max_workers,
            pool_maxsize=self.config.search_max_workers * 2,
            max_retries=0  # 재시도는 우리 코드에서 처리
        )
        self._search_session.mount('http://', adapter)
        self._search_session.mount('https://', adapter)

        # =========================================================================
        # [Phase 6] 완전 비동기 스트리밍 Reward 지원
        # - 백그라운드 스레드에서 Frozen Generator + Reward 처리
        # - 메인 루프 블로킹 없이 GPU 100% 활용
        # =========================================================================
        self._pending_threads: List[threading.Thread] = []  # 완료 대기 중인 스레드
        self._thread_lock = threading.Lock()  # Thread-safety를 위한 Lock
        self.generated_answers: Dict[int, str] = {}  # 생성된 답변 저장소
        self._streaming_frozen_generated: set = set()  # 스트리밍에서 처리 완료된 샘플 인덱스

        # =========================================================================
        # [Phase 7] Tool 호출 완전 비동기화
        # - Search API 호출을 비동기로 시작하고, bbox/search_complete와 병렬 처리
        # - Search 결과는 필요한 시점에만 대기
        # - GPU idle 시간 최소화
        # =========================================================================
        self._tool_executor = ThreadPoolExecutor(
            max_workers=getattr(config, 'search_max_workers', 8),
            thread_name_prefix="ToolAsync"
        )
        self._phase7_enabled = getattr(config, 'phase7_tool_async', True)  # Phase 7 활성화 플래그

    def _ensure_saves_complete(self) -> int:
        """
        [Phase 4] 모든 비동기 이미지 저장 완료 대기

        Returns:
            int: 완료된 저장 작업 수
        """
        if not self._pending_saves:
            return 0

        completed = 0
        errors = []
        for future in self._pending_saves:
            try:
                future.result(timeout=30)  # 30초 타임아웃
                completed += 1
            except Exception as e:
                errors.append(str(e))

        self._pending_saves.clear()

        if errors:
            print(f"[Phase 4] Image save errors ({len(errors)}): {errors[:3]}...")

        return completed

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']
    
    def _postprocess_responses_first(self,batch):
        
        responses_str = self.tokenizer.batch_decode(batch.batch['input_ids'], skip_special_tokens=True)
        responses_str = ["<search>"+item.split('Question: ')[1].split(' \n\nassistant\n')[0]+"</search>" for item in responses_str]

        responses = self._batch_tokenize(responses_str)
        return responses, responses_str
        

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        def extract_tags(text):
            # [Phase 1] 사전 컴파일된 정규식 사용 (성능 최적화)
            matches = _RE_EXTRACT_TAGS.findall(text)
            result = "\n".join([f"<{tag}>{content}</{tag}>" for tag, content in matches])
            return result

        responses_str = [extract_tags(resp) + self.tokenizer.eos_token for resp in responses_str]

        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    # def _process_next_obs(self, next_obs: List, rollings) -> torch.Tensor:
    #     """Process next observations from environment."""
    #     next_obs_str = []
    #     multi_modal_data = []
    #     multi_modal_inputs = []
    #     merge_length = self.processor.image_processor.merge_size**2
    #     # print(self.retrievaled_images)
    #     for idx, obs_item in enumerate(next_obs):
    #         # invalid
    #         if isinstance(obs_item,str):
    #             next_obs_str.append(obs_item)
    #             multi_modal_data.append({'image': []})
    #             multi_modal_inputs.append(BatchFeature(dict()))
    #         # invalid
    #         elif isinstance(obs_item, list) and not isinstance(obs_item[0],dict) and len(self.retrievaled_images[idx]) == 0:
    #             next_obs_str.append('\n<|im_start|>user\nYour previous action is invalid. You must conduct reasoning inside <think> and <think> every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine using <search> query </search> and the user will return the search results. Whenever you retrieve an image, you may crop it for a clearer view using <bbox>[x1, y1, x2, y2]</bbox>. You can search as many times as you want. If you determine that no further knowledge is needed, you must finish with <search_complete>true</search_complete>. Otherwise, continue with <search> or <bbox> actions until you are ready to finish. Please try again.\n<|im_end|>\n<|im_start|>assistant\n')
    #             multi_modal_data.append({'image': []})
    #             multi_modal_inputs.append(BatchFeature(dict()))
    #         # crop
    #         elif isinstance(obs_item,list) and not isinstance(obs_item[0],dict):
    #             try:
    #                 latest_image = rollings.non_tensor_batch['multi_modal_data'][idx]['image'][-1]
    #                 width, height = latest_image.size
    #                 raw_images_crop = Image.open(self.retrievaled_images[idx][-1])
    #                 raw_width, raw_height = raw_images_crop.size
    #                 if self.is_validation:
    #                     obs_item = [obs_item[0]-28, obs_item[1]-28, obs_item[2]+28, obs_item[3]+28]
    #                 crop_area = [int(raw_width * obs_item[0] / width), int(raw_height * obs_item[1] / height), int(raw_width * obs_item[2] / width), int(raw_height * obs_item[3] / height)]
    #                 crop_area = [max(0, crop_area[0]), max(0, crop_area[1]), min(raw_width, crop_area[2]), min(raw_height, crop_area[3])]
    #                 input_images_list = [raw_images_crop.crop((crop_area[0], crop_area[1], crop_area[2], crop_area[3]))]
    #                 raw_images_list = [process_image(image, 512*28*28, 256*28*28) for image in input_images_list]

    #                 #generator added
    #                 crop_path = os.path.join(self.config.crops_dir, f"{uuid.uuid4().hex}.jpg")
    #                 raw_images_list[0].save(crop_path)
    #                 self.cropped_images[idx].append(crop_path)
    #                 #                    

    #                 multi_modal_data.append({'image': raw_images_list})
    #                 image_inputs = self.processor.image_processor(raw_images_list, return_tensors='pt') 
    #                 multi_modal_inputs.append(image_inputs)
    #                 image_grid_thw = image_inputs['image_grid_thw']
    #                 obs_str = ''.join([f"<|vision_start|>{self.processor.image_token * (image_grid_thw_item.prod() // merge_length)}<|vision_end|>" for image_grid_thw_item in image_grid_thw])
    #                 raw_obs_str = f"<|vision_start|>{self.processor.image_token}<|vision_end|>" * len(image_grid_thw) 
    #                 obs_str = '\n<|im_start|>user\n' + obs_str + '<|im_end|>\n<|im_start|>assistant\n'
    #                 next_obs_str.append(obs_str)   
    #             except Exception as e:
    #                 next_obs_str.append('\n<|im_start|>user\nYour previous action is invalid. You must conduct reasoning inside <think> and </think> every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine using <search> query </search> and the user will return the search results. Whenever you retrieve an image, you may crop it for a clearer view using <bbox>[x1, y1, x2, y2]</bbox>. You can search as many times as you want. If you determine that no further external knowledge is needed, you must finish with <search_complete>true</search_complete>. Otherwise, continue with <search> or <bbox> actions until you are ready to finish. Please try again.\n<|im_end|>\n<|im_start|>assistant\n')
    #                 multi_modal_data.append({'image': []})
    #                 multi_modal_inputs.append(BatchFeature(dict())) 
    #         # ret image
    #         elif isinstance(obs_item,list) and isinstance(obs_item[0],dict):

    #             img_file_list = [item['image_file'] for item in obs_item]
    #             for image_item in img_file_list:
    #                 if image_item not in self.retrievaled_images[idx]:
    #                     self.retrievaled_images[idx].append(image_item)
    #                     # input_images_list = img_file_list[:1]
    #                     input_images_list = [image_item]
    #                     break
    #             #수정 pixe_value
    #             # raw_images_list = [process_image(image, 512*28*28, 256*28*28) for image in input_images_list]

    #             # multi_modal_data.append({'image': raw_images_list})
    #             # image_inputs = self.processor.image_processor(raw_images_list, return_tensors='pt')

    #             # multi_modal_inputs.append(image_inputs)
    #             # image_grid_thw = image_inputs['image_grid_thw']

    #             # obs_str = ''.join([f"<|vision_start|>{self.processor.image_token * (image_grid_thw_item.prod() // merge_length)}<|vision_end|>" for image_grid_thw_item in image_grid_thw])
    #             # raw_obs_str = f"<|vision_start|>{self.processor.image_token}<|vision_end|>" * len(image_grid_thw) 
    #             # obs_str = '\n<|im_start|>user\n' + obs_str + '<|im_end|>\n<|im_start|>assistant\n'
    #             # next_obs_str.append(obs_str)
    #             raw_images_list = [process_image(image, 512*28*28, 256*28*28) for image in input_images_list]
    #             image_inputs = self.processor.image_processor(raw_images_list, return_tensors='pt')

    #             if 'pixel_values' in image_inputs:
    #                 # 정상인 경우: 기존 로직 수행
    #                 multi_modal_data.append({'image': raw_images_list})
    #                 multi_modal_inputs.append(image_inputs)
                    
    #                 image_grid_thw = image_inputs['image_grid_thw']
    #                 obs_str = ''.join([f"<|vision_start|>{self.processor.image_token * (image_grid_thw_item.prod() // merge_length)}<|vision_end|>" for image_grid_thw_item in image_grid_thw])
    #                 obs_str = '\n<|im_start|>user\n' + obs_str + '<|im_end|>\n<|im_start|>assistant\n'
    #                 next_obs_str.append(obs_str)
    #             else:
    #                 # 실패한 경우: System Error 메시지 삽입 및 빈 데이터 추가
    #                 print(f"Warning: Image processing failed for idx {idx}. Inserting Error Message.")
    #                 multi_modal_data.append({'image': []})
    #                 multi_modal_inputs.append(BatchFeature(dict()))
                    
    #                 error_msg = "\n<|im_start|>user\nSystem Error: Failed to load image due to format issues.\n<|im_end|>\n<|im_start|>assistant\n"
    #                 next_obs_str.append(error_msg)
    #             #//
    #         else:
    #             raise ValueError('invalid observation')
    
    
        # next_obs_ids = self.tokenizer(
        #     next_obs_str, 
        #     padding='longest',
        #     return_tensors='pt',
        #     add_special_tokens=False,  # Prevents adding special tokens
        # )['input_ids']

        # return next_obs_ids, next_obs_str, multi_modal_data, multi_modal_inputs
    def _process_next_obs(self, next_obs: List, rollings) -> torch.Tensor:
            """Process next observations from environment."""
            next_obs_str = []
            multi_modal_data = []
            multi_modal_inputs = []
            merge_length = self.processor.image_processor.merge_size**2
            
            for idx, obs_item in enumerate(next_obs):
                # 1. Invalid String
                if isinstance(obs_item,str):
                    next_obs_str.append(obs_item)
                    multi_modal_data.append({'image': []})
                    multi_modal_inputs.append(BatchFeature(dict()))

                # 2. Invalid Action (No previous image)
                elif isinstance(obs_item, list) and not isinstance(obs_item[0],dict) and len(self.retrievaled_images[idx]) == 0:
                    next_obs_str.append('\n<|im_start|>user\nInvalid action: No image to crop. Please search first.\n<|im_end|>\n<|im_start|>assistant\n')
                    multi_modal_data.append({'image': []})
                    multi_modal_inputs.append(BatchFeature(dict()))

                # 3. [BBOX / CROP] 구간
                elif isinstance(obs_item,list) and not isinstance(obs_item[0],dict):
                    try:
                        # 기존 로직 수행
                        latest_image = rollings.non_tensor_batch['multi_modal_data'][idx]['image'][-1]
                        width, height = latest_image.size
                        # [Phase 4] LRU 캐시 적용: 동일 이미지 반복 로딩 시 디스크 I/O 회피
                        raw_images_crop = _cached_image_open(self.retrievaled_images[idx][-1])
                        raw_width, raw_height = raw_images_crop.size

                        if self.is_validation:
                            obs_item = [obs_item[0]-28, obs_item[1]-28, obs_item[2]+28, obs_item[3]+28]
                        crop_area = [int(raw_width * obs_item[0] / width), int(raw_height * obs_item[1] / height), int(raw_width * obs_item[2] / width), int(raw_height * obs_item[3] / height)]
                        crop_area = [max(0, crop_area[0]), max(0, crop_area[1]), min(raw_width, crop_area[2]), min(raw_height, crop_area[3])]
                        input_images_list = [raw_images_crop.crop((crop_area[0], crop_area[1], crop_area[2], crop_area[3]))]
                        raw_images_list = [process_image(image, 512*28*28, 256*28*28) for image in input_images_list]

                        # generator added
                        # [Phase 4] 비동기 이미지 저장: 블로킹 I/O를 백그라운드로 이동
                        crop_path = os.path.join(self.config.crops_dir, f"{uuid.uuid4().hex}.jpg")
                        # 이미지 복사 후 비동기 저장 (원본 객체 수정 방지)
                        img_to_save = raw_images_list[0].copy()
                        future = self._save_executor.submit(img_to_save.save, crop_path)
                        self._pending_saves.append(future)
                        self.cropped_images[idx].append(crop_path)

                        image_inputs = self.processor.image_processor(raw_images_list, return_tensors='pt')

                        # [검증] pixel_values 확인
                        if 'pixel_values' in image_inputs:
                            multi_modal_data.append({'image': raw_images_list})
                            multi_modal_inputs.append(image_inputs)
                            image_grid_thw = image_inputs['image_grid_thw']
                            obs_str = ''.join([f"<|vision_start|>{self.processor.image_token * (image_grid_thw_item.prod() // merge_length)}<|vision_end|>" for image_grid_thw_item in image_grid_thw])
                            raw_obs_str = f"<|vision_start|>{self.processor.image_token}<|vision_end|>" * len(image_grid_thw)
                            obs_str = '\n<|im_start|>user\n' + obs_str + '<|im_end|>\n<|im_start|>assistant\n'
                            next_obs_str.append(obs_str)
                        else:
                            raise ValueError("BBox processing produced no pixel_values")

                    except Exception as e:
                        # [BBOX 실패 시 명확한 에러 메시지]
                        print(f"[DEBUG] Bbox Error at idx {idx}: {e}")
                        next_obs_str.append('\n<|im_start|>user\n[System Error: Bbox Crop Failed] The image crop operation failed. Please try a different action.\n<|im_end|>\n<|im_start|>assistant\n')
                        multi_modal_data.append({'image': []})
                        multi_modal_inputs.append(BatchFeature(dict()))

                # 4. [SEARCH / RETRIEVAL] 구간
                elif isinstance(obs_item,list) and isinstance(obs_item[0],dict):

                    img_file_list = [item['image_file'] for item in obs_item]
                    for image_item in img_file_list:
                        if image_item not in self.retrievaled_images[idx]:
                            self.retrievaled_images[idx].append(image_item)
                            input_images_list = [image_item]
                            break
                    
                    try:
                        raw_images_list = [process_image(image, 512*28*28, 256*28*28) for image in input_images_list]
                        image_inputs = self.processor.image_processor(raw_images_list, return_tensors='pt')

                        if 'pixel_values' in image_inputs:
                            multi_modal_data.append({'image': raw_images_list})
                            multi_modal_inputs.append(image_inputs)
                            
                            image_grid_thw = image_inputs['image_grid_thw']
                            obs_str = ''.join([f"<|vision_start|>{self.processor.image_token * (image_grid_thw_item.prod() // merge_length)}<|vision_end|>" for image_grid_thw_item in image_grid_thw])
                            obs_str = '\n<|im_start|>user\n' + obs_str + '<|im_end|>\n<|im_start|>assistant\n'
                            next_obs_str.append(obs_str)
                        else:
                            # [SEARCH 실패 시 명확한 에러 메시지]
                            print(f"[DEBUG] Search Image Error at idx {idx}: No pixel_values")
                            error_msg = "\n<|im_start|>user\n[System Error: Search Image Failed] The retrieved image is corrupted or invalid.\n<|im_end|>\n<|im_start|>assistant\n"
                            next_obs_str.append(error_msg)
                            multi_modal_data.append({'image': []})
                            multi_modal_inputs.append(BatchFeature(dict()))

                    except Exception as e:
                        print(f"[DEBUG] Search Processing Exception at idx {idx}: {e}")
                        error_msg = "\n<|im_start|>user\n[System Error: Search Image Processing Exception]\n<|im_end|>\n<|im_start|>assistant\n"
                        next_obs_str.append(error_msg)
                        multi_modal_data.append({'image': []})
                        multi_modal_inputs.append(BatchFeature(dict()))

                else:
                    raise ValueError('invalid observation')
            
            next_obs_ids = self.tokenizer(
                next_obs_str, 
                padding='longest',
                return_tensors='pt',
                add_special_tokens=False,
            )['input_ids']

            return next_obs_ids, next_obs_str, multi_modal_data, multi_modal_inputs
#//

    # def _concat_multi_modal_data(self, rollings, next_obs_multi_modal_data:list, next_obs_multi_modal_inputs:list):
    #     if not 'multi_modal_inputs' in rollings.non_tensor_batch.keys():

    #         rollings.non_tensor_batch['multi_modal_inputs'] = np.empty(len(next_obs_multi_modal_data), dtype=object)
    #         for idx, item in enumerate(next_obs_multi_modal_inputs):
    #             rollings.non_tensor_batch['multi_modal_inputs'][idx] = item

    #         rollings.non_tensor_batch['multi_modal_data'] = np.array(next_obs_multi_modal_data, dtype=object)

    #     else:
    #         for idx, multi_modal_data_item in enumerate(next_obs_multi_modal_data):
    #             if len(multi_modal_data_item['image']) > 0:
    #                 # data
    #                 #수정 pixel_value
    #                 # rollings.non_tensor_batch['multi_modal_data'][idx]['image'].extend(multi_modal_data_item['image'])
    #                 # if 'pixel_values' in rollings.non_tensor_batch['multi_modal_inputs'][idx]:
    #                 #     rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'] = torch.cat((rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'], next_obs_multi_modal_inputs[idx]['pixel_values']),dim=0)
    #                 #     rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'] = torch.cat((rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'], next_obs_multi_modal_inputs[idx]['image_grid_thw']),dim=0)
    #                 # else:
    #                 #     rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'] = next_obs_multi_modal_inputs[idx]['pixel_values']
    #                 #     rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'] = next_obs_multi_modal_inputs[idx]['image_grid_thw']
    #                 # ▼▼▼ [수정 핵심] 이미지가 있다고 해도, 실제 텐서 키(pixel_values)가 있는지 한 번 더 확인해야 함 ▼▼▼
    #                 rollings.non_tensor_batch['multi_modal_data'][idx]['image'].extend(multi_modal_data_item['image'])
                    
    #                 # [핵심 수정] pixel_values가 있으면 정상 병합, 없으면 "더미 데이터" 생성하여 병합
    #                 if 'pixel_values' in next_obs_multi_modal_inputs[idx]:
    #                     # A. 정상 케이스
    #                     if 'pixel_values' in rollings.non_tensor_batch['multi_modal_inputs'][idx]:
    #                         rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'] = torch.cat((rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'], next_obs_multi_modal_inputs[idx]['pixel_values']),dim=0)
    #                         rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'] = torch.cat((rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'], next_obs_multi_modal_inputs[idx]['image_grid_thw']),dim=0)
    #                     else:
    #                         rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'] = next_obs_multi_modal_inputs[idx]['pixel_values']
    #                         rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'] = next_obs_multi_modal_inputs[idx]['image_grid_thw']
                    
    #                 else:
    #                     # B. 비정상 케이스 (토큰은 있는데 픽셀값이 사라짐) -> 더미 데이터 주입하여 짝 맞춤
    #                     print(f"Warning: 'pixel_values' missing at idx {idx} in _concat. Using Dummy Black Image to prevent IndexError.")
                        
    #                     # Qwen2-VL 기준 더미 데이터 생성 (1x1 픽셀)
    #                     # pixel_values: 대략적인 shape과 타입만 맞추면 됨
    #                     dummy_pixel_values = torch.zeros((1, 1176), dtype=torch.float32).to(rollings.batch['input_ids'].device) # 1176 is minimal flattened size roughly
                        
    #                     # image_grid_thw: [1, h, w] -> [1, 1, 1] (시간1, 높이1, 너비1)
    #                     dummy_grid = torch.tensor([[1, 1, 1]], dtype=torch.long).to(rollings.batch['input_ids'].device)

    #                     if 'pixel_values' in rollings.non_tensor_batch['multi_modal_inputs'][idx]:
    #                         # 기존 텐서와 모양(Shape)을 맞춰서 병합 시도 (실패 시 안전하게 unsqueeze 등 처리)
    #                         try:
    #                             # 기존 pixel_values의 feature dimension(마지막 차원)을 확인하여 맞춤
    #                             expected_dim = rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'].shape[-1]
    #                             if dummy_pixel_values.shape[-1] != expected_dim:
    #                                  dummy_pixel_values = torch.zeros((1, expected_dim), dtype=torch.float32).to(rollings.batch['input_ids'].device)

    #                             rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'] = torch.cat((
    #                                 rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'], 
    #                                 dummy_pixel_values
    #                             ), dim=0)
    #                             rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'] = torch.cat((
    #                                 rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'], 
    #                                 dummy_grid
    #                             ), dim=0)
    #                         except Exception as e:
    #                             print(f"Error merging dummy tensor: {e}. Skipping (Crash risk high).")
    #                     else:
    #                         # 초기 할당
    #                         rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'] = dummy_pixel_values
    #                         rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'] = dummy_grid
    #                 # ▲▲▲ [수정 끝] ▲▲▲
    #     return rollings
    def _concat_multi_modal_data(self, rollings, next_obs_multi_modal_data: list, next_obs_multi_modal_inputs: list):
        """
        [Phase 2 최적화] 롤링 상태에 멀티모달 데이터를 연결합니다.

        최적화 내용:
        - 반복적인 딕셔너리 접근을 로컬 변수로 캐싱
        - 조건문 구조 개선으로 불필요한 체크 감소
        - 성능 로깅: Phase2PerfTimer로 측정 가능 (PHASE2_PERF_LOG=1)
        """
        with Phase2PerfTimer("_concat_multi_modal_data", batch_size=len(next_obs_multi_modal_data)):

            # [Phase 2] 자주 접근하는 딕셔너리를 로컬 변수로 캐싱
            non_tensor_batch = rollings.non_tensor_batch

            if 'multi_modal_inputs' not in non_tensor_batch:
                # 초기화 케이스
                non_tensor_batch['multi_modal_inputs'] = np.empty(len(next_obs_multi_modal_data), dtype=object)
                for idx, item in enumerate(next_obs_multi_modal_inputs):
                    non_tensor_batch['multi_modal_inputs'][idx] = item

                non_tensor_batch['multi_modal_data'] = np.array(next_obs_multi_modal_data, dtype=object)

            else:
                # [Phase 2] 기존 데이터를 로컬 변수로 캐싱
                existing_multi_modal_data = non_tensor_batch['multi_modal_data']
                existing_multi_modal_inputs = non_tensor_batch['multi_modal_inputs']

                for idx, multi_modal_data_item in enumerate(next_obs_multi_modal_data):
                    # [Phase 2] 이미지가 없으면 조기 종료 (불필요한 체크 회피)
                    if len(multi_modal_data_item['image']) == 0:
                        continue

                    new_inputs = next_obs_multi_modal_inputs[idx]

                    # 방어 로직: pixel_values가 있을 때만 병합
                    if 'pixel_values' not in new_inputs:
                        continue

                    # [Phase 2] 로컬 변수 캐싱으로 딕셔너리 접근 횟수 감소
                    existing_inputs = existing_multi_modal_inputs[idx]
                    existing_data = existing_multi_modal_data[idx]

                    # 이미지 리스트 확장
                    existing_data['image'].extend(multi_modal_data_item['image'])

                    # 텐서 연결
                    if 'pixel_values' in existing_inputs:
                        # [Phase 2] 기존 텐서와 새 텐서 연결
                        existing_inputs['pixel_values'] = torch.cat(
                            (existing_inputs['pixel_values'], new_inputs['pixel_values']),
                            dim=0
                        )
                        existing_inputs['image_grid_thw'] = torch.cat(
                            (existing_inputs['image_grid_thw'], new_inputs['image_grid_thw']),
                            dim=0
                        )
                    else:
                        # 첫 번째 이미지인 경우 직접 할당
                        existing_inputs['pixel_values'] = new_inputs['pixel_values']
                        existing_inputs['image_grid_thw'] = new_inputs['image_grid_thw']

            return rollings
#//

    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        if next_obs_ids.shape[1] != 0:
            new_input_ids = self.tensor_fn.concatenate_with_padding([
                rollings.batch['input_ids'],
                cur_responses,
                next_obs_ids
            ])
        else:
            new_input_ids = self.tensor_fn.concatenate_with_padding([
                rollings.batch['input_ids'],
                cur_responses
            ])
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        }, rollings.non_tensor_batch)

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None and next_obs_ids.shape[1] != 0:
            responses = self.tensor_fn.concatenate_with_padding([
                right_side['responses'],
                cur_responses,
                next_obs_ids
            ], pad_to_left=False)
        else:
            responses = self.tensor_fn.concatenate_with_padding([
                right_side['responses'],
                cur_responses,
            ], pad_to_left=False)
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len]}


    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
        [Phase 2 최적화] Wrapper for generation that handles multi-GPU padding requirements.

        최적화 내용:
        - 텐서 연산 통합: 3회 torch.cat → 더 효율적인 구조로 개선
        - non_tensor_batch 처리: 중첩 루프 제거, 리스트 곱셈 활용
        - 성능 로깅: Phase2PerfTimer로 측정 가능 (PHASE2_PERF_LOG=1)
        """
        with Phase2PerfTimer("_generate_with_gpu_padding",
                            batch_size=active_batch.batch['input_ids'].shape[0] if active_batch.batch is not None else None):

            num_gpus = self.config.num_gpus
            if num_gpus <= 1:
                return self.actor_rollout_wg.generate_sequences(active_batch)

            batch_size = active_batch.batch['input_ids'].shape[0]
            remainder = batch_size % num_gpus

            if remainder == 0:
                return self.actor_rollout_wg.generate_sequences(active_batch)

            # Add padding sequences
            padding_size = num_gpus - remainder

            # [Phase 2] 패딩 ID 토크나이징 (1회만 수행)
            padded_ids = self.tokenizer(
                ['<|im_start|>user\nHi, who are u?<|im_end|>\n<|im_start|>assistant\n'],
                padding='longest',
                return_tensors='pt',
                add_special_tokens=False,
            )['input_ids'][0]  # 바로 [0] 인덱싱

            # [Phase 2] 패딩 텐서 생성
            pad_input_ids = torch.full_like(active_batch.batch['input_ids'][0], 151643, dtype=torch.int64)
            pad_input_ids[:len(padded_ids)] = padded_ids
            pad_attention_mask = self.tensor_fn.create_attention_mask(pad_input_ids)
            pad_input_ids = pad_input_ids.unsqueeze(0)
            pad_attention_mask = pad_attention_mask.unsqueeze(0)
            pad_position_ids = self.tensor_fn.create_position_ids(pad_attention_mask)

            # [Phase 2] 텐서 배치 구성 - 반복 횟수 사전 계산
            # repeat 인자를 미리 계산하여 재사용
            repeat_dims_2d = (padding_size, 1)  # 2D 텐서용

            padded_batch = {
                'attention_mask': torch.cat([
                    active_batch.batch['attention_mask'],
                    pad_attention_mask.repeat(*repeat_dims_2d)
                ], dim=0),
                'input_ids': torch.cat([
                    active_batch.batch['input_ids'],
                    pad_input_ids.repeat(*repeat_dims_2d)
                ], dim=0),
                'position_ids': torch.cat([
                    active_batch.batch['position_ids'],
                    pad_position_ids.repeat(*repeat_dims_2d)
                ], dim=0),
            }

            # [Phase 2] Non-tensor batch 처리 최적화
            # - 중첩 루프 제거: 리스트 곱셈으로 대체
            # - 각 키별 처리를 인라인화
            padded_non_tensor_batch = {}
            list_ids = padded_ids.tolist()  # 1회만 변환

            for k, v in active_batch.non_tensor_batch.items():
                if k == 'raw_prompt_ids':
                    # 리스트 곱셈으로 동일한 객체 참조 (메모리 효율)
                    pad_items = [list_ids] * padding_size
                elif k == 'multi_modal_inputs':
                    # 각각 새로운 딕셔너리 생성 (mutable 객체이므로)
                    pad_items = [{} for _ in range(padding_size)]
                elif k == 'multi_modal_data':
                    pad_items = [{'image': []} for _ in range(padding_size)]
                else:
                    # 알 수 없는 키: None으로 패딩
                    pad_items = [None] * padding_size

                # [Fix] 차원 불일치 방지: 리스트로 변환 후 concatenate
                # v가 다차원일 수 있으므로 tolist()로 변환 후 합침
                if isinstance(v, np.ndarray):
                    combined = list(v) + pad_items
                else:
                    combined = list(v) + pad_items
                padded_non_tensor_batch[k] = np.array(combined, dtype=object)

            padded_active_batch = DataProto.from_dict(padded_batch, padded_non_tensor_batch)

            # Generate with padded batch
            padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

            # [Phase 2] Remove padding from output - 딕셔너리 컴프리헨션 유지 (이미 효율적)
            trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}

            # Handle meta_info if present
            if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
                trimmed_meta = {}
                for k, v in padded_output.meta_info.items():
                    if isinstance(v, torch.Tensor):
                        trimmed_meta[k] = v[:-padding_size]
                    else:
                        trimmed_meta[k] = v
                padded_output.meta_info = trimmed_meta

            padded_output.batch = trimmed_batch
            return padded_output

    def _raw_prompt_ids(self, rollings):
        """
        [Phase 2 최적화] 롤링 상태의 input_ids에서 연속된 이미지 토큰(151655)을 압축합니다.

        최적화 내용:
        - 내부 함수를 메서드 레벨로 이동하여 매번 정의하는 오버헤드 제거
        - 텐서 연산 최적화: torch 마스킹 활용
        - 성능 로깅: Phase2PerfTimer로 측정 가능 (PHASE2_PERF_LOG=1)
        """
        with Phase2PerfTimer("_raw_prompt_ids", batch_size=rollings.batch['input_ids'].shape[0]):

            # [Phase 2] long() 변환
            input_ids = rollings.batch['input_ids'].long()
            attention_mask = rollings.batch['attention_mask']

            # [Phase 2] 배치 처리를 위한 텐서 마스킹 활용
            # 각 샘플별로 유효한 토큰만 추출
            batch_size = input_ids.shape[0]
            raw_next_obs_ids = []

            for idx in range(batch_size):
                # 마스크가 1인 위치의 토큰만 추출
                valid_ids = input_ids[idx][attention_mask[idx] == 1].tolist()

                # 연속된 이미지 토큰 압축 (인라인 처리)
                compressed = self._compress_consecutive_tokens(valid_ids, 151655)
                raw_next_obs_ids.append(compressed)

            raw_next_obs_ids = np.array(raw_next_obs_ids, dtype=object)
            rollings.non_tensor_batch['raw_prompt_ids'] = raw_next_obs_ids
            rollings.batch['input_ids'] = input_ids  # long() 변환된 버전 저장

            return rollings

    def _compress_consecutive_tokens(self, arr: list, target: int) -> list:
        """
        [Phase 2] 연속된 target 토큰을 하나로 압축합니다.

        Args:
            arr: 토큰 ID 리스트
            target: 압축할 대상 토큰 (예: 151655 = 이미지 토큰)

        Returns:
            압축된 토큰 리스트
        """
        if not arr:
            return arr

        result = []
        i = 0
        n = len(arr)

        while i < n:
            if arr[i] == target:
                result.append(target)
                # 연속된 target 건너뛰기
                while i + 1 < n and arr[i + 1] == target:
                    i += 1
            else:
                result.append(arr[i])
            i += 1

        return result

    def deactivate_batch(self, active_mask,rollings):
        raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids']
        max_model_len = 10240
        curr_active_mask = torch.tensor([len(raw_prompt_ids_item) < max_model_len for raw_prompt_ids_item in raw_prompt_ids], dtype=torch.bool)
        active_mask = active_mask * curr_active_mask
        return active_mask

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""

        meta_info = {}

        # [NEW] 스트리밍 모드 초기화
        if self.streaming_reward_manager:
            self._init_prompt_tracking(gen_batch)

        # [Phase 6] 완전 비동기 스트리밍 상태 초기화
        self._pending_threads.clear()
        self.generated_answers.clear()
        self._streaming_frozen_generated.clear()

        # ▼▼▼[성능 측정 추가] 1. 로그 파일 및 모니터 객체 초기화▼▼▼ 수정
        # 고유한 로그 파일 이름을 생성하여 모든 측정 결과를 한 파일에 기록합니다.
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"./logs/generation_detail_{current_time}_{uuid.uuid4().hex[:6]}.txt"
        
        # 측정 지점 1: 메인 모델(Actor)의 '계획' 생성 성능 측정용
        actor_monitor = GPUMonitor(log_file=log_filename, label="[1] Actor Generation (Planning)")
        
        # 측정 지점 2: 외부 도구(검색 API) 호출 시간 측정용
        tool_monitor = GPUMonitor(log_file=log_filename, label="[2] Tool Execution (Search API)")
        
        # 측정 지점 3: Frozen 모델의 '최종 답변' 생성 성능 측정용
        frozen_monitor = GPUMonitor(log_file=log_filename, label="[3] Frozen Generator (Answering)")
        # ▲▲▲[성능 측정 추가]▲▲▲        

        original_left_side = {'input_ids': initial_input_ids}
        original_right_side = {'responses': initial_input_ids[:, []]}

        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids']

        #generator added
        self.search_completed = [False] * gen_batch.batch['input_ids'].shape[0]

        # ===== (4) 첫 턴에서 질문 문자열 저장(원래 파싱 방식) & 컨테이너 준비 =====
        decoded_inputs = self.tokenizer.batch_decode(initial_input_ids, skip_special_tokens=True)
        '''
        최종 generator에게 초반 쿼리를 넘겨주기 위해서.
        '''
        self.questions = []
        for s in decoded_inputs:
            try:
                q = s.split('Question: ')[1].split(' \n\nassistant\n')[0]
            except Exception:
                q = s  # fallback
            self.questions.append(q)
        #


        self.retrievaled_images = [[] for _ in range(gen_batch.batch['input_ids'].shape[0])]
        self.cropped_images = [[] for _ in range(gen_batch.batch['input_ids'].shape[0])]      # generator added

        # [Phase 3] 루프 불변값 추출: max_turns - 1 계산을 루프 외부로 이동
        last_turn_idx = self.config.max_turns - 1

        ############======================🚀Main generation loop🚀==================######################
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            # [Phase 3] 루프 불변 상수 사용
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=_CUT_KEYS
            ) #데이터 압축

            rollings = self._raw_prompt_ids(rollings)#전처리 

            active_mask = self.deactivate_batch(active_mask, rollings) #최대 길이를 넘으면 deactivate
            if not active_mask.sum():
                break
            
            #수정 제거 max turn 5
            # if 'multi_modal_inputs' in rollings.non_tensor_batch.keys():
            #     rollings_active = DataProto.from_dict(
            #         tensors={k: v[active_mask] for k, v in rollings.batch.items()},
            #         non_tensors={k: v[active_mask] for k, v in rollings.non_tensor_batch.items()}
            #     )
            # else:
            #     rollings_active = DataProto.from_dict({
            #         k: v[active_mask] for k, v in rollings.batch.items()
            #     })  
            
            # [Phase 3] 루프 불변값 사용: last_turn_idx
            is_last_turn = step == last_turn_idx

            if not is_last_turn:
                # [Phase 3] .keys() 제거: in dict가 더 효율적
                if 'multi_modal_inputs' in rollings.non_tensor_batch:
                    rollings_active = DataProto.from_dict(
                        tensors={k: v[active_mask] for k, v in rollings.batch.items()},
                        non_tensors={k: v[active_mask] for k, v in rollings.non_tensor_batch.items()}
                    )
                else:
                    rollings_active = DataProto.from_dict({
                        k: v[active_mask] for k, v in rollings.batch.items()
                    })
            #


            #수정 maxturn 5
            # actor_monitor.start() #측정 지점 1: '계획' 생성 성능 측정 수정
            # gen_output = self._generate_with_gpu_padding(rollings_active)
            # actor_monitor.stop() #측정 끝
                actor_monitor.start() #측정 지점 1: '계획' 생성 성능 측정 수정
                gen_output = self._generate_with_gpu_padding(rollings_active)
                actor_monitor.stop() #측정 끝            
            #//    

            #수정 max turn 5
            #meta_info = gen_output.meta_info     
                meta_info = gen_output.meta_info
            #//

            #수정 mac turn5
            # responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            # print(responses_str[0])
                responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
                print(responses_str[0])
            else:
                forced_count = active_mask.sum().item()
                responses_str = [FORCED_COMPLETION_RESPONSE] * forced_count
                if forced_count > 0:
                    responses_ids = self._batch_tokenize(responses_str)
                else:
                    responses_ids = torch.empty((0, 0), dtype=rollings.batch['input_ids'].dtype)
            #//            


            
            # Execute in environment and process observations
            
            #개별 예제(example) 수준에서 빈자리를 채워주는(pad)'
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            #수정 추가 max turn 5
            responses_ids = responses_ids.to(rollings.batch['input_ids'].device)
            #//


            #수정----#
            # 1. execute_predictions를 호출하기 전에 uids를 가져옵니다

            all_uids = rollings.non_tensor_batch['id']


            # 2. Execute in environment and process observations
            #    호출 시 uids를 두 번째 인자로 전달합니다.

            tool_monitor.start() #'행동'을 위한 외부 도구 호출 시간 측정▼▼▼ 수정
            next_obs, dones = self.execute_predictions(responses_str, all_uids, self.tokenizer.pad_token, active_mask)
            tool_monitor.stop() #측정 끝

            # --- 여기까지 ---

            #next_obs, dones = self.execute_predictions(responses_str, self.tokenizer.pad_token, active_mask) #수정 제거 uid 넘기기
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            next_obs_ids, next_obs_str, next_obs_multi_modal_data, next_obs_multi_modal_inputs = self._process_next_obs(next_obs, rollings)
            
            rollings = self._concat_multi_modal_data(
                rollings,
                next_obs_multi_modal_data,
                next_obs_multi_modal_inputs
            )
            
            # Update states            
            rollings = self._update_rolling_state(
                rollings,
                responses_ids, #수정 제거 
                #padded_responses_ids, #수정 추가 uid
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids, #수정 제거 uid
                #padded_responses_ids, #수정 추가 uid
                next_obs_ids
            )



        # final LLM rollout
        if active_mask.sum():

            # [Phase 3] 루프 불변 상수 사용
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=_CUT_KEYS
            )

            rollings = self._raw_prompt_ids(rollings)

            active_mask = self.deactivate_batch(active_mask, rollings)

            if active_mask.sum():
                # [Phase 3] .keys() 제거: in dict가 더 효율적
                if 'multi_modal_inputs' in rollings.non_tensor_batch:
                    rollings_active = DataProto.from_dict(
                        tensors={k: v[active_mask] for k, v in rollings.batch.items()},
                        non_tensors={k: v[active_mask] for k, v in rollings.non_tensor_batch.items()}
                    )
                else:
                    rollings_active = DataProto.from_dict({
                        k: v[active_mask] for k, v in rollings.batch.items()
                    })

                gen_output = self._generate_with_gpu_padding(rollings_active)

                meta_info = gen_output.meta_info
                responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
                responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

                all_uids = rollings.non_tensor_batch['id'] #수정 uid 추가 


                # # Execute in environment and process observations
                _, dones = self.execute_predictions( #ctive uid 추가 수정
                    responses_str, all_uids, self.tokenizer.pad_token, active_mask, do_search=False
                )

                curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
                active_mask = active_mask * curr_active_mask
                active_num_list.append(active_mask.sum().item())

                original_right_side = self._update_right_side(
                    original_right_side,
                    responses_ids,
                )
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        # =================== raw prompt ids ===================
        rollings.non_tensor_batch['raw_prompt_ids'] = raw_prompt_ids
        # rollings.non_tensor_batch.pop('raw_prompt_ids')
        
        if not self.is_validation:
            rollings, original_right_side = self._add_noisy_multi_modal_data(rollings, original_right_side)
        ### check again
        
        retrievaled_images_array = np.empty(len(self.retrievaled_images), dtype=object)
        for idx in range(len(self.retrievaled_images)):
            retrievaled_images_array[idx] = self.retrievaled_images[idx]
        rollings.non_tensor_batch['retrievaled_images'] = retrievaled_images_array
        # ===== generator added=====
        gen_to_tokenize = [""] * len(self.retrievaled_images)

        # =========================================================================
        # [Phase 6] 백그라운드 스레드 완료 대기
        # - 스트리밍 모드에서 백그라운드로 처리 중인 Frozen Generator 완료 대기
        # =========================================================================
        if self._pending_threads:
            pending_count = len(self._pending_threads)
            print(f"[Phase 6] 백그라운드 스레드 대기 중... ({pending_count}개)")
            wait_start = _time.perf_counter()

            for thread in self._pending_threads:
                thread.join(timeout=120)  # 최대 2분 대기

            wait_elapsed = _time.perf_counter() - wait_start
            print(f"[Phase 6] 백그라운드 스레드 완료: {pending_count}개, {wait_elapsed:.2f}초")
            self._pending_threads.clear()

        # [Phase 6] 스트리밍에서 이미 처리된 샘플 건너뛰기
        # - 스트리밍 모드에서 이미 Frozen Generator가 호출된 샘플은 제외
        completed_indices = [
            i for i, flag in enumerate(self.search_completed)
            if flag and i not in self._streaming_frozen_generated
        ]

        if completed_indices:
            batch_questions = []
            batch_paths = []

            for i in completed_indices:
                q = self.questions[i]
                paths = self._prepare_generator_images(self.retrievaled_images[i], self.cropped_images[i])
                batch_questions.append(q)
                batch_paths.append(paths)

            frozen_monitor.start()
            index2answer = self._call_frozen_generator_batch(
                completed_indices, batch_questions, batch_paths
            )
            frozen_monitor.stop()

            # 새로 생성된 답변 저장
            for i in completed_indices:
                ans = index2answer.get(i, "")
                self.generated_answers[i] = ans

        # [Phase 6] 모든 완료된 샘플의 답변 적용 (스트리밍 + 배치 모두 포함)
        for i, flag in enumerate(self.search_completed):
            if flag:
                ans = self.generated_answers.get(i, "")
                if ans:
                    gen_to_tokenize[i] = f"<answer>{ans}</answer>{self.tokenizer.eos_token}"

        ans_ids = self.tokenizer(
            gen_to_tokenize, padding='longest', return_tensors='pt', add_special_tokens=False
        )['input_ids']

        original_right_side = self._update_right_side(original_right_side, ans_ids)
        rollings = self._update_rolling_state(
            rollings, ans_ids, next_obs_ids=torch.zeros((ans_ids.shape[0], 0), dtype=torch.long)
        )

        # [Phase 4] 모든 비동기 이미지 저장 완료 대기
        saved_count = self._ensure_saves_complete()
        if saved_count > 0:
            print(f"[Phase 4] Async image saves completed: {saved_count}")

        return self._compose_final_output(original_left_side, original_right_side, meta_info, rollings)
    
    def _add_noisy_multi_modal_data(self, rollings, original_right_side):
        image_padded = Image.new('RGB', (64, 64), (0, 0, 0))

        image_padded = process_image(image_padded, 256*256, 128*128)
        image_inputs = self.processor.image_processor([image_padded], return_tensors='pt')
        image_grid_thw = image_inputs['image_grid_thw']
        merge_length = self.processor.image_processor.merge_size**2
        padded_str = f"\n<|im_start|>user\n<|vision_start|>{self.processor.image_token * (image_grid_thw.prod() // merge_length)}<|vision_end|><|im_end|>"

        padded_str_list = []
        for idx, multi_modal_item in enumerate(rollings.non_tensor_batch['multi_modal_data']):
            if len(multi_modal_item['image']) == 0:
                padded_str_list.append(padded_str)
                rollings.non_tensor_batch['multi_modal_data'][idx]['image'].append(image_padded)
                rollings.non_tensor_batch['multi_modal_inputs'][idx] = image_inputs
            else:
                padded_str_list.append('')
            
        padded_ids = self.tokenizer(
            padded_str_list, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        original_right_side = self._update_right_side(
            original_right_side,
            padded_ids
        )
        return rollings, original_right_side


    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict,
                            rollings) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )

        final_output = DataProto.from_dict(final_output,rollings.non_tensor_batch)
        final_output.meta_info.update(meta_info)
        
        return final_output

    # =========================================================================
    # [NEW] 비동기 병렬 검색 메서드
    # =========================================================================
    def _search_single_batch(self, batch_reqs: List[Dict], max_retries: int = 3) -> List[Dict]:
        """
        단일 배치 검색 요청 (워커 스레드에서 실행)
        실패 시 지수 백오프로 재시도

        최적화:
        - HTTP 연결 풀링 사용 (self._search_session)
        - config에서 timeout 설정 (기본 60초)
        """
        last_error = None
        timeout = self.config.search_timeout  # 기본값 60초

        for attempt in range(max_retries):
            try:
                # [최적화] Session 사용으로 연결 재사용
                response = self._search_session.post(
                    self.config.search_url,
                    json=batch_reqs,
                    timeout=timeout
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                last_error = e
                wait_time = (2 ** attempt) + _random.uniform(0, 1)
                print(f"[Search] 배치 검색 오류 (시도 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    _time.sleep(wait_time)

        # 모든 재시도 실패 시 예외 발생
        raise RuntimeError(f"검색 배치 {max_retries}회 재시도 실패: {last_error}")

    def _async_search_batches(self, search_requests: List[Dict]) -> Dict[int, List]:
        """
        비동기 병렬 검색 - ThreadPoolExecutor로 배치들을 병렬 처리

        Args:
            search_requests: 검색 요청 리스트 [{query, id, request_idx}, ...]

        Returns:
            request_idx -> results 매핑

        Raises:
            RuntimeError: 검색 실패 시 (재시도 후에도 실패)
        """
        if not search_requests:
            return {}

        batch_size = self.config.search_batch_size
        max_workers = self.config.search_max_workers

        # 배치로 분할
        batches = [
            search_requests[i:i + batch_size]
            for i in range(0, len(search_requests), batch_size)
        ]

        all_results = []
        errors = []

        # 병렬 실행
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._search_single_batch, batch): idx
                for idx, batch in enumerate(batches)
            }

            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    errors.append((batch_idx, str(e)))

        # 오류가 있으면 예외 발생
        if errors:
            error_msg = "; ".join([f"배치{idx}: {err}" for idx, err in errors])
            raise RuntimeError(f"병렬 검색 실패: {error_msg}")

        # 결과 매핑 생성
        results_map = {
            item['request_idx']: item.get('results', [])
            for item in all_results
        }

        return results_map

    # execute_predictions 함수
    def execute_predictions(self, predictions: List[str], uids: np.ndarray, pad_token: str, active_mask=None, do_search=True) -> List[str]:
        """
        [Phase 7] Tool 호출 완전 비동기화

        핵심 변경:
        1. Search API 호출을 비동기로 즉시 시작 (블로킹 없음)
        2. bbox, search_complete를 search와 병렬로 처리
        3. Search 결과는 마지막에 대기

        이로 인해 bbox 처리 시간과 search API 대기 시간이 중첩됩니다.
        """
        cur_actions, contents = self.postprocess_predictions(predictions)

        # [Phase 7] 결과 리스트를 미리 초기화 (인덱스로 접근하기 위해)
        n_samples = len(cur_actions)
        next_obs = [None] * n_samples
        dones = [None] * n_samples

        # [Phase 3] deque 사용: pop(0) O(n) → popleft() O(1)
        bbox_queue = deque(content for action, content in zip(cur_actions, contents) if action == 'bbox')

        # =========================================================================
        # [Phase 7] Step 1: Search 요청을 비동기로 즉시 시작 (블로킹 없음!)
        # =========================================================================
        search_requests = []
        search_indices = []  # search 액션의 원래 인덱스 저장
        for i, (action, content) in enumerate(zip(cur_actions, contents)):
            if action == 'search':
                # [Phase 1] 사전 컴파일된 정규식 사용 (성능 최적화)
                m = _RE_UID_SUFFIX.search(str(uids[i]))
                search_id = int(m.group(1)) if m else -1

                search_requests.append({
                    "query": content,
                    "id": str(search_id),
                    "request_idx": i
                })
                search_indices.append(i)

        # [Phase 7] Search 비동기 시작 (Future 저장, 대기하지 않음!)
        search_future = None
        if do_search and len(search_requests) > 0 and self._phase7_enabled:
            # 비동기 호출 시작 (ThreadPoolExecutor 사용)
            search_future = self._tool_executor.submit(
                self._async_search_batches, search_requests
            )
            # 이 시점에서 search API 호출이 백그라운드에서 진행 중!

        # =========================================================================
        # [Phase 7] Step 2: bbox, search_complete 등 즉시 처리 (search와 병렬!)
        # =========================================================================
        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            if not active:
                next_obs[i] = ''
                dones[i] = 1
            elif action == 'search':
                # [Phase 7] Search 결과는 나중에 채움 (Step 3에서)
                dones[i] = 0  # done 값은 먼저 설정
                # next_obs[i]는 Step 3에서 설정
            elif action == 'bbox':
                try:
                    bbox_value = json.loads(bbox_queue.popleft())
                    if len(bbox_value) == 4 and bbox_value[0] >= 0 and bbox_value[1] >= 0 and bbox_value[2] >= 0 and bbox_value[3] >= 0:
                        next_obs[i] = bbox_value
                    else:
                        raise ValueError("Invalid bbox value")
                except:
                    # [Phase 3] 상수 사용
                    next_obs[i] = _MSG_INVALID_BBOX
                dones[i] = 0
            elif action == 'search_complete':
                is_true = contents[i].strip().lower() == 'true'
                if is_true:
                    self.search_completed[i] = True

                    # [NEW] 스트리밍 Reward: 프롬프트 완료 체크
                    if self.streaming_reward_manager:
                        self._check_and_submit_prompt_reward(i)

                next_obs[i] = ''
                dones[i] = 1  # trajectory 종료
            else:
                # [Phase 3] 상수 사용
                next_obs[i] = _MSG_INVALID_ACTION
                dones[i] = 0

        # =========================================================================
        # [Phase 7] Step 3: Search 결과 대기 및 반영 (마지막에!)
        # =========================================================================
        if do_search and len(search_requests) > 0:
            if self._phase7_enabled and search_future is not None:
                # [Phase 7] 비동기 결과 대기 (이 시점에 bbox 처리는 이미 완료!)
                try:
                    results_map = search_future.result(timeout=60)  # 60초 타임아웃
                except Exception as e:
                    print(f"[Phase 7] Search 비동기 호출 실패: {e}")
                    results_map = {i: [] for i in search_indices}  # 빈 결과로 폴백
            else:
                # Phase 7 비활성화 시 기존 방식 사용
                if getattr(self.config, 'async_search', True):
                    results_map = self._async_search_batches(search_requests)
                else:
                    batch_size = getattr(self.config, 'search_batch_size', 100)
                    search_results_list = []
                    for i in range(0, len(search_requests), batch_size):
                        batch_reqs = search_requests[i:i + batch_size]
                        response = requests.post(self.config.search_url, json=batch_reqs)
                        search_results_single_batch = response.json()
                        search_results_list.extend(search_results_single_batch)
                    results_map = {item['request_idx']: item.get('results', []) for item in search_results_list}

            # Search 결과를 next_obs에 반영
            for i in search_indices:
                if active_mask[i]:
                    next_obs[i] = results_map.get(i, [])

        # [Phase 7] 혹시 None인 항목이 있으면 빈 값으로 채움 (안전장치)
        for i in range(n_samples):
            if next_obs[i] is None:
                next_obs[i] = []
            if dones[i] is None:
                dones[i] = 0

        return next_obs, dones


    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output

                #수정 max turn 5
                # pattern = r'<(search|bbox|search_complete)>(.*?)</\1>'
                # match = re.search(pattern, prediction, re.DOTALL)
                # if match:
                #     content = match.group(2).strip()  # Return only the content inside the tags
                #     action = match.group(1)
                stripped_prediction = prediction.strip()
                if stripped_prediction == FORCED_COMPLETION_RESPONSE:
                    content = 'true'
                    action = 'search_complete'
                #//

                else:
                    # [Phase 1] 사전 컴파일된 정규식 사용 (성능 최적화)
                    match = _RE_ACTION_PATTERN.search(prediction)
                    if match:
                        content = match.group(2).strip()  # Return only the content inside the tags
                        action = match.group(1)
                    else:
                        content = ''
                        action = None                    
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    #generator added
    # ===== (8) generator 이미지 준비 =====
    def _prepare_generator_images(self, originals: List[str], crops: List[str]) -> List[str]:
        # 존재하는 파일만, 중복 제거, 최대 장수 제한
        seen = set()
        out = []
        for p in (originals + crops):
            if p and (p not in seen) and os.path.exists(p):
                seen.add(p)
                out.append(p)
            if len(out) >= self.config.generator_max_images:
                break
        return out



    def _call_frozen_generator_single(self, question: str, image_paths: List[str]) -> Tuple[int, str]:
        if not _HAS_DASHSCOPE:
            print("🚨 오류: DashScope 설정이 누락되었습니다. (.env 키 확인 또는 라이브러리 설치 필요)") # 디버깅
            return (0, "")

        try:
            # 빈 프롬프트 방지(400 회피)
            qtext = (question or "").strip() or "."

            sys_prompt = (
                "You are a visual QA generator. "
                "Use only the provided images and the user question. "
                "Return ONLY the final answer text without extra explanations."
            )

            # 이미지 파트 구성 (file:// 강제)
            user_content = []
            if image_paths:
                for p in image_paths:
                    part = _to_image_part(p)  # >>> ADDED: helper 사용
                    if part:
                        user_content.append(part)
            user_content.append({"text": f"Question: {qtext}"})

            messages = []
            if getattr(self.config, "use_system_prompt", True):
                messages.append({"role": "system", "content": [{"text": sys_prompt}]})
            messages.append({"role": "user", "content": user_content})

            try:
                resp = _dashscope_call_with_fallback(
                    model=self.config.frozen_model,
                    messages=messages,
                    max_tokens=int(getattr(self.config, "frozen_max_tokens", 256)),
                )
            except Exception:
                print(f"🚨 [API ERROR] Question: {question[:30]}... | Error: {e}")  # 디버깅
                return (0, "")

            code = getattr(resp, "status_code", None)
            if code == HTTPStatus.OK:
                text = _extract_text_from_multimodal(resp) or ""
                return (200, text)
            
            return (int(code) if isinstance(code, HTTPStatus) else (code or 0), "")
        except Exception:
            print(f"🚨 오류: API 호출 중 에러 발생: {e}") # 디버깅
            return (0, "")


    def _call_frozen_generator_batch(
        self,
        indices: List[int],
        questions: List[str],
        images_list: List[List[str]],
    ) -> Dict[int, str]:
        """
        Frozen Generator 배치 호출 (동기 인터페이스)

        내부적으로 OpenAI AsyncClient가 사용 가능하면 비동기 처리,
        그렇지 않으면 기존 DashScope SDK 동기 방식으로 폴백합니다.
        """
        results: Dict[int, str] = {}
        if not indices:
            return results

        # [Phase 5] OpenAI AsyncClient 사용 가능시 비동기 처리
        if _HAS_OPENAI_ASYNC and _OPENAI_ASYNC_CLIENT is not None:
            return self._call_frozen_generator_batch_async_wrapper(
                indices, questions, images_list
            )

        # === 폴백: 기존 DashScope SDK 동기 방식 ===
        return self._call_frozen_generator_batch_sync(
            indices, questions, images_list
        )

    def _call_frozen_generator_batch_sync(
        self,
        indices: List[int],
        questions: List[str],
        images_list: List[List[str]],
    ) -> Dict[int, str]:
        """기존 DashScope SDK 동기 방식 (폴백용)"""
        results: Dict[int, str] = {}
        if not indices:
            return results

        workers = max(1, int(getattr(self.config, "generator_batch_workers", 4)))
        workers = min(workers, 4)
        max_retries = int(getattr(self.config, "frozen_max_retries", 3))
        backoff_base = float(getattr(self.config, "frozen_backoff_base", 1.5))

        def _once_with_retry(idx: int, q: str, paths: List[str]) -> Tuple[int, str]:
            delay = 0.0
            for attempt in range(max_retries):
                if delay > 0:
                    _time.sleep(delay)
                code, ans = self._call_frozen_generator_single(q, paths)

                if code == 200:
                    if ans:
                        return idx, ans
                    else:
                        return idx, ""

                if code in (429, 500, 502, 503, 504, 0):
                    delay = (backoff_base ** attempt) + _random.uniform(0, 0.2)
                    continue

                return idx, ""

            return idx, ""

        for start in range(0, len(indices), workers):
            end = start + workers
            chunk_idx = indices[start:end]
            chunk_q = questions[start:end]
            chunk_img = images_list[start:end]

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_once_with_retry, i, q, p) for i, q, p in zip(chunk_idx, chunk_q, chunk_img)]
                for f in as_completed(futs):
                    try:
                        i, ans = f.result()
                    except Exception:
                        i, ans = None, ""
                    if i is not None:
                        results[i] = ans or ""

            _time.sleep(0.05)

        return results

    def _call_frozen_generator_batch_async_wrapper(
        self,
        indices: List[int],
        questions: List[str],
        images_list: List[List[str]],
    ) -> Dict[int, str]:
        """
        [Phase 5] 비동기 배치 처리를 동기 인터페이스로 래핑

        asyncio.run() 또는 기존 이벤트 루프에서 실행합니다.
        """
        try:
            # 기존 이벤트 루프가 있는지 확인
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None:
                # 이미 이벤트 루프 안에 있는 경우 (Ray 워커 등)
                # nest_asyncio 또는 ThreadPoolExecutor 사용
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._call_frozen_generator_batch_async(
                            indices, questions, images_list
                        )
                    )
                    return future.result(timeout=300)  # 5분 타임아웃
            else:
                # 이벤트 루프가 없는 경우 직접 실행
                return asyncio.run(
                    self._call_frozen_generator_batch_async(
                        indices, questions, images_list
                    )
                )
        except Exception as e:
            _phase2_logger.warning(f"[Phase5] Async batch failed, falling back to sync: {e}")
            return self._call_frozen_generator_batch_sync(indices, questions, images_list)

    async def _call_frozen_generator_batch_async(
        self,
        indices: List[int],
        questions: List[str],
        images_list: List[List[str]],
    ) -> Dict[int, str]:
        """
        [Phase 5] OpenAI AsyncClient를 사용한 비동기 배치 처리

        - asyncio.gather()로 전체 배치를 한 번에 병렬 처리
        - 세마포어로 동시 요청 수 제한 (rate limit 대응)
        - 재시도 로직 포함
        """
        results: Dict[int, str] = {}
        if not indices:
            return results

        # 설정값 가져오기
        max_concurrent = int(getattr(self.config, "frozen_max_concurrent", 50))
        max_retries = int(getattr(self.config, "frozen_max_retries", 3))
        backoff_base = float(getattr(self.config, "frozen_backoff_base", 1.5))
        max_tokens = int(getattr(self.config, "frozen_max_tokens", 1024))
        model = getattr(self.config, "frozen_model", "qwen2.5-vl-72b-instruct")

        # 동시 요청 수 제한용 세마포어
        semaphore = asyncio.Semaphore(max_concurrent)

        start_time = _time.perf_counter()
        _phase2_logger.info(
            f"[Phase5] Starting async batch: {len(indices)} samples, "
            f"max_concurrent={max_concurrent}"
        )

        async def _single_with_retry(idx: int, q: str, paths: List[str]) -> Tuple[int, str]:
            """재시도 로직이 포함된 단일 비동기 호출"""
            delay = 0.0
            for attempt in range(max_retries):
                if delay > 0:
                    await asyncio.sleep(delay)

                code, ans = await _call_frozen_generator_async_single(
                    client=_OPENAI_ASYNC_CLIENT,
                    model=model,
                    question=q,
                    image_paths=paths,
                    max_tokens=max_tokens,
                    semaphore=semaphore,
                )

                if code == 200:
                    return (idx, ans if ans else "")

                # 재시도 가능한 오류
                if code in (429, 500, 502, 503, 504, 408, 0):
                    delay = (backoff_base ** attempt) + _random.uniform(0, 0.3)
                    continue

                # 기타 오류는 빈 결과 반환
                return (idx, "")

            return (idx, "")

        # 모든 요청을 병렬로 실행
        tasks = [
            _single_with_retry(idx, q, paths)
            for idx, q, paths in zip(indices, questions, images_list)
        ]

        # asyncio.gather로 전체 배치 동시 처리
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 수집
        success_count = 0
        for result in task_results:
            if isinstance(result, Exception):
                _phase2_logger.warning(f"[Phase5] Task exception: {result}")
                continue
            if isinstance(result, tuple) and len(result) == 2:
                idx, ans = result
                results[idx] = ans or ""
                if ans:
                    success_count += 1

        elapsed = _time.perf_counter() - start_time
        _phase2_logger.info(
            f"[Phase5] Async batch completed: {success_count}/{len(indices)} success, "
            f"elapsed={elapsed:.2f}s, "
            f"throughput={len(indices)/elapsed:.1f} req/s"
        )

        return results

    # =========================================================================
    # 스트리밍 Reward 관련 메서드들
    # =========================================================================
    def _init_prompt_tracking(self, gen_batch):
        """
        프롬프트별 완료 추적 초기화

        n_agent 구조에서 각 프롬프트의 샘플들을 그룹화하여 추적합니다.
        프롬프트의 모든 샘플이 완료되면 Reward 계산을 시작합니다.
        """
        uids = gen_batch.non_tensor_batch.get('uid', gen_batch.non_tensor_batch.get('id', []))
        n_agent = getattr(self.config, 'n_agent', 8)  # 기본값 8

        batch_size = len(uids)
        num_prompts = batch_size // n_agent

        self._prompt_completion_status.clear()

        for prompt_idx in range(num_prompts):
            base_idx = prompt_idx * n_agent
            # uid에서 고유 프롬프트 ID 추출 (n_agent 샘플들은 같은 베이스 uid를 공유)
            uid = str(uids[base_idx])
            # uid에서 마지막 숫자 부분 제거하여 프롬프트 ID 생성
            prompt_id = uid.rsplit('_', 1)[0] if '_' in uid else uid

            self._prompt_completion_status[prompt_id] = {
                'total_samples': n_agent,
                'completed_samples': 0,
                'sample_indices': list(range(base_idx, base_idx + n_agent)),
                'submitted': False
            }

        print(f"[Generation] 스트리밍 추적 초기화: {num_prompts}개 프롬프트, "
              f"각 {n_agent}개 샘플")

    def _check_and_submit_prompt_reward(self, sample_idx: int):
        """
        [Phase 6] 샘플 완료 시 프롬프트 전체 완료 여부 확인 후 백그라운드에서 처리

        완전 비동기 방식:
        1. 프롬프트의 모든 샘플이 완료되면 백그라운드 스레드 시작
        2. 백그라운드에서 Frozen Generator 호출 → Reward 제출
        3. 메인 루프는 블로킹 없이 계속 진행

        Args:
            sample_idx: 완료된 샘플의 배치 내 인덱스
        """
        n_agent = getattr(self.config, 'n_agent', 8)
        prompt_idx = sample_idx // n_agent

        # 프롬프트 ID 찾기
        prompt_ids = list(self._prompt_completion_status.keys())
        if prompt_idx >= len(prompt_ids):
            return

        prompt_id = prompt_ids[prompt_idx]
        status = self._prompt_completion_status.get(prompt_id)
        if not status or status['submitted']:
            return

        status['completed_samples'] += 1

        # 프롬프트의 모든 샘플이 완료되었는지 확인
        if status['completed_samples'] >= status['total_samples']:
            # [Phase 6] 백그라운드 스레드로 처리 (블로킹 없음!)
            indices = list(status['sample_indices'])  # 복사본 생성
            thread = threading.Thread(
                target=self._process_prompt_background,
                args=(indices, prompt_id, status),
                daemon=True,
                name=f"FrozenGen-{prompt_id}"
            )
            thread.start()
            self._pending_threads.append(thread)
            status['submitted'] = True  # 중복 제출 방지

            print(f"[Phase 6] 프롬프트 {prompt_id} 백그라운드 처리 시작 "
                  f"(샘플 {len(indices)}개)")

    def _process_prompt_background(self, indices: List[int], prompt_id: str, status: dict):
        """
        [Phase 6] 백그라운드 스레드에서 Frozen Generator + Reward 처리

        이 함수는 별도 스레드에서 실행되어 메인 루프를 블로킹하지 않습니다.

        Args:
            indices: 처리할 샘플 인덱스 리스트
            prompt_id: 프롬프트 ID
            status: 프롬프트 상태 딕셔너리
        """
        try:
            start_time = _time.perf_counter()

            # 1. Frozen Generator 호출 준비
            batch_questions = []
            batch_paths = []

            for i in indices:
                q = self.questions[i] if i < len(self.questions) else ""
                paths = self._prepare_generator_images(
                    self.retrievaled_images[i] if i < len(self.retrievaled_images) else [],
                    self.cropped_images[i] if i < len(self.cropped_images) else []
                )
                batch_questions.append(q)
                batch_paths.append(paths)

            # 2. Frozen Generator 호출 (Phase 5 비동기 배치 처리)
            index2answer = self._call_frozen_generator_batch(
                indices, batch_questions, batch_paths
            )

            # 3. 결과 저장 (Thread-safe)
            with self._thread_lock:
                for i in indices:
                    answer = index2answer.get(i, "")
                    self.generated_answers[i] = answer
                    self._streaming_frozen_generated.add(i)

            frozen_elapsed = _time.perf_counter() - start_time

            # 4. samples_data 수집 (generated_answer 포함)
            samples_data = self._collect_samples_data(indices)

            # 5. Reward 제출 (Gemini VLM Judge 호출)
            self.streaming_reward_manager.submit_prompt(
                uid=prompt_id,
                sample_indices=indices,
                samples_data=samples_data
            )

            total_elapsed = _time.perf_counter() - start_time
            success_count = sum(1 for i in indices if self.generated_answers.get(i))

            print(f"[Phase 6] 프롬프트 {prompt_id} 완료: "
                  f"Frozen={frozen_elapsed:.2f}s, Total={total_elapsed:.2f}s, "
                  f"Success={success_count}/{len(indices)}")

        except Exception as e:
            print(f"[Phase 6] 프롬프트 {prompt_id} 처리 실패: {e}")
            import traceback
            traceback.print_exc()

    def _collect_samples_data(self, indices: List[int]) -> List[Dict]:
        """
        [Phase 6] Reward 계산에 필요한 샘플 데이터 수집

        스트리밍 모드에서 RMManager에 전달할 데이터를 수집합니다.
        Phase 6에서 generated_answer 필드가 추가되어 Gemini VLM Judge가
        올바르게 평가할 수 있습니다.

        Args:
            indices: 수집할 샘플 인덱스 리스트

        Returns:
            각 샘플의 전처리된 데이터 리스트
        """
        samples_data = []

        for idx in indices:
            # 검색된 이미지 경로
            retrieved_images = list(self.retrievaled_images[idx]) if idx < len(self.retrievaled_images) else []

            # NDCG 계산용 basename 추출
            retrieved_basenames = [
                os.path.basename(p.rstrip('/')).split(".jpg")[0]
                for p in retrieved_images
            ]

            # 질문 추출
            question = self.questions[idx] if idx < len(self.questions) else ''

            # [Phase 6] Frozen Generator에서 생성된 답변
            generated_answer = self.generated_answers.get(idx, '') if hasattr(self, 'generated_answers') else ''

            samples_data.append({
                'query': question,
                'retrieved_images': retrieved_images,
                'retrieved_basenames': retrieved_basenames,
                'generated_answer': generated_answer,  # [Phase 6] NEW!
                # 아래 필드들은 나중에 ray_trainer.py에서 채워질 예정
                'response_str': '',
                'reference_answer': '',
                'reference_image_paths': [],
                'reference_basenames': [],
            })

        return samples_data



