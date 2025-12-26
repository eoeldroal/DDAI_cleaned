"""
SearchEngine - GPU 적극 활용 최적화 버전 v2

핵심 최적화:
1. 사전 패딩된 청크 텐서: 초기화 시 1회 패딩 → 쿼리 시 패딩 불필요
2. GPU 상주 텐서: 모든 이미지 임베딩이 GPU에 연속 메모리로 상주
3. 초대형 배치: A100 80GB 메모리를 최대한 활용
4. 벡터화된 연산: einsum 직접 수행 (기존과 수학적으로 100% 동일)

입출력 결과는 기존과 100% 동일합니다.
기존 버전: search_engine_deprecated.py
"""
import os
import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Mapping, Any, Dict, Tuple
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
import hashlib
from collections import OrderedDict
import threading

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo, ImageNode
from vl_embedding import VL_Embedding


# =========================================================================
# GPU 적극 활용 설정 (A100 80GB 최적화)
# =========================================================================
class GPUConfig:
    """GPU 적극 활용 설정 - A100 80GB 기준 최대 성능"""

    # 사전 패딩 청크 크기 (이미지 임베딩을 이 단위로 미리 패딩)
    # 큰 값일수록 메모리 사용량 증가, 연산 효율 향상
    PREPAD_CHUNK_SIZE = 16384        # 16K 이미지씩 청크로 사전 패딩

    # 스코어링 배치 크기 - 매우 적극적으로!
    QUERY_BATCH_SIZE = 256           # 한 번에 처리할 쿼리 수
    IMAGE_CHUNK_BATCH = 32768        # 한 번에 처리할 이미지 수 (32K)

    # 메모리 관리 (GPU 적극 사용을 위해 간격 늘림)
    GC_INTERVAL = 500                # 500개 요청마다 GC (자주 안 함)
    CACHE_CLEAR_INTERVAL = 2000      # 2000개 요청마다 GPU 캐시 클리어

    # 쿼리 캐시 설정
    QUERY_CACHE_SIZE = 100000        # 10만개로 증가


# =========================================================================
# 쿼리 결과 LRU 캐시
# =========================================================================
class QueryResultCache:
    """Thread-safe LRU 캐시 (쿼리 결과 캐싱)"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: OrderedDict[str, List[str]] = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _hash_query(self, query: str) -> str:
        """쿼리 문자열을 해시로 변환"""
        return hashlib.md5(query.encode('utf-8')).hexdigest()

    def get(self, query: str) -> Optional[List[str]]:
        """캐시에서 결과 조회"""
        key = self._hash_query(query)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)  # LRU 갱신
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None

    def put(self, query: str, result: List[str]):
        """캐시에 결과 저장"""
        key = self._hash_query(query)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)  # 가장 오래된 항목 제거
                self.cache[key] = result

    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': f"{hit_rate:.2%}"
            }

    def clear(self):
        """캐시 초기화"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0


def nodefile2node(input_file):
    nodes = []
    for doc in json.load(open(input_file, 'r')):
        if doc['class_name'] == 'TextNode' and doc['text'] != '':
            nodes.append(TextNode.from_dict(doc))
        elif doc['class_name'] == 'ImageNode':
            nodes.append(ImageNode.from_dict(doc))
        else:
            continue
    return nodes


class SearchEngine:
    """
    GPU 적극 활용 검색 엔진

    핵심 최적화:
    1. 사전 패딩된 청크 텐서: 초기화 시 1회 패딩 → 쿼리 시 0회
    2. GPU 상주: 모든 이미지 임베딩이 GPU에 연속 메모리로 저장
    3. 벡터화된 einsum: 기존과 수학적으로 100% 동일한 결과
    4. 초대형 배치: A100 80GB 메모리 최대 활용

    입출력 결과는 기존과 100% 동일합니다.

    Args:
        dataset_dir: 코퍼스 디렉토리 경로
        node_dir_prefix: 노드 디렉토리 접두사
        embed_model_name: 임베딩 모델 이름
        cache_size: 쿼리 캐시 크기 (기본값: 100,000)
        enable_cache: 캐시 활성화 여부
        gpu_id: 사용할 GPU ID (기본값: 2)
        chunk_size: 사전 패딩 청크 크기 (기본값: 16,384)
    """

    def __init__(
        self,
        dataset_dir='search_engine/corpus',
        node_dir_prefix='colqwen_ingestion',
        embed_model_name='vidore/colqwen2-v1.0',
        cache_size: int = None,
        enable_cache: bool = True,
        gpu_id: int = 2,
        chunk_size: int = None,
        # 하위 호환성을 위해 유지 (무시됨)
        score_batch_size: int = None,
    ):
        self.workers = 1
        self.dataset_dir = dataset_dir
        self.node_dir = os.path.join(self.dataset_dir, node_dir_prefix)

        # GPU 설정
        self.gpu_id = gpu_id

        # 청크 크기 설정 (사전 패딩 단위)
        if chunk_size is not None:
            GPUConfig.PREPAD_CHUNK_SIZE = chunk_size

        print(f"[SearchEngine] GPU: cuda:{self.gpu_id}")
        print(f"[SearchEngine] 사전 패딩 청크 크기: {GPUConfig.PREPAD_CHUNK_SIZE:,}")

        # 임베딩 모델 초기화 (지정된 GPU)
        self.vector_embed_model = VL_Embedding(
            model=embed_model_name,
            mode='image',
            device=f'cuda:{self.gpu_id}'
        )

        # 캐시 초기화
        cache_size = cache_size or GPUConfig.QUERY_CACHE_SIZE
        self.enable_cache = enable_cache
        self.query_cache = QueryResultCache(max_size=cache_size) if enable_cache else None

        # 쿼리 엔진 로드
        self.query_engine = self.load_query_engine()

        # 메모리 관리 카운터
        self._request_count = 0

        # 통계
        self._total_queries = 0
        self._total_time = 0.0

        if enable_cache:
            print(f"[SearchEngine] 쿼리 캐시 활성화 (max_size={cache_size})")

        print(f"[SearchEngine] 초기화 완료 - {self.image_nums:,}개 이미지 로드됨")

    def load_nodes(self):
        files = os.listdir(self.node_dir)
        parsed_files = []
        max_workers = 10
        if max_workers == 1:
            for file in tqdm(files):
                input_file = os.path.join(self.node_dir, file)
                suffix = input_file.split('.')[-1]
                if suffix != 'node':
                    continue
                nodes = nodefile2node(input_file)
                parsed_files.extend(nodes)
        else:
            def parse_file(file, node_dir):
                input_file = os.path.join(node_dir, file)
                suffix = input_file.split('.')[-1]
                if suffix != 'node':
                    return []
                return nodefile2node(input_file)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(tqdm(executor.map(parse_file, files, [self.node_dir]*len(files)), total=len(files)))
            for result in results:
                parsed_files.extend(result)
        return parsed_files

    def load_query_engine(self):
        """
        노드 및 임베딩 로드 (GPU 적극 활용 버전)

        최적화:
        1. 사전 패딩된 청크 텐서 생성 (초기화 시 1회만)
        2. GPU에 연속 메모리로 저장
        3. 매 쿼리마다 pad_sequence 호출 불필요
        """
        print('Loading nodes...')
        self.nodes = self.load_nodes()

        device = self.vector_embed_model.embed_model.device
        print(f"[SearchEngine] 임베딩을 {device}로 사전 패딩하여 로드 중...")

        # Step 1: 모든 임베딩을 CPU에서 생성 (개별 텐서 리스트)
        raw_embeddings = [
            torch.tensor(node.embedding).view(-1, 128).bfloat16()
            for node in tqdm(self.nodes, desc="Creating Embeddings")
        ]
        self.image_nums = len(raw_embeddings)

        # Step 2: 청크별로 사전 패딩하여 GPU에 저장
        # 기존 score_multi_vector와 동일한 padding_value=0 사용
        chunk_size = GPUConfig.PREPAD_CHUNK_SIZE
        self.prepadded_chunks: List[torch.Tensor] = []
        self.chunk_sizes: List[int] = []  # 각 청크의 실제 이미지 수

        print(f"[SearchEngine] 청크 크기: {chunk_size:,}, 총 청크 수: {(self.image_nums + chunk_size - 1) // chunk_size}")

        for i in tqdm(range(0, self.image_nums, chunk_size), desc="Pre-padding chunks"):
            chunk = raw_embeddings[i:i + chunk_size]
            actual_size = len(chunk)
            self.chunk_sizes.append(actual_size)

            # 청크 내 텐서들을 패딩하여 하나의 3D 텐서로 변환
            # [chunk_size, max_seq_len, 128]
            # padding_value=0: 기존 score_multi_vector와 동일!
            padded_chunk = torch.nn.utils.rnn.pad_sequence(
                chunk, batch_first=True, padding_value=0
            ).to(device)

            self.prepadded_chunks.append(padded_chunk)

        # 원본 리스트 삭제 (메모리 절약)
        del raw_embeddings
        gc.collect()

        # 호환성을 위해 embedding_img도 유지 (None으로 - 사용하지 않음)
        self.embedding_img = None

        # GPU 메모리 상태 출력
        torch.cuda.empty_cache()
        mem_used = torch.cuda.memory_allocated(self.gpu_id) / 1024**3
        mem_total = torch.cuda.get_device_properties(self.gpu_id).total_memory / 1024**3
        print(f"[SearchEngine] GPU {self.gpu_id}: {mem_used:.2f}GB / {mem_total:.2f}GB 사용 중")
        print(f"[SearchEngine] 사전 패딩 완료: {len(self.prepadded_chunks)}개 청크, "
              f"총 {self.image_nums:,}개 이미지")

    def load_node_postprocessors(self):
        return []

    def _maybe_cleanup_memory(self):
        """주기적 메모리 정리 (GPU 적극 사용을 위해 간격 늘림)"""
        self._request_count += 1

        # GC 수행 (간격 늘림)
        if self._request_count % GPUConfig.GC_INTERVAL == 0:
            gc.collect()

        # GPU 캐시 클리어 (간격 늘림)
        if self._request_count % GPUConfig.CACHE_CLEAR_INTERVAL == 0:
            torch.cuda.empty_cache()
            print(f"[SearchEngine] 메모리 정리 완료 (요청 #{self._request_count})")

    def _score_prepadded_chunks(
        self,
        query_embeddings: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        사전 패딩된 청크에 대해 MaxSim 스코어 계산 (GPU 적극 활용)

        수학적으로 기존 score_multi_vector와 100% 동일:
        einsum("bnd,csd->bcns", q, p).max(dim=3)[0].sum(dim=2)

        최적화:
        1. 사전 패딩된 청크 사용 → pad_sequence 호출 불필요
        2. 전체 계산이 GPU에서 수행
        3. 최종 결과만 필요시 CPU로 이동

        Args:
            query_embeddings: 쿼리 임베딩 리스트 또는 텐서
            device: 계산 디바이스

        Returns:
            [num_queries, num_images] 스코어 텐서
        """
        # 쿼리 임베딩이 리스트인 경우 패딩
        if isinstance(query_embeddings, list):
            qs_padded = torch.nn.utils.rnn.pad_sequence(
                query_embeddings, batch_first=True, padding_value=0
            ).to(device)
        else:
            qs_padded = query_embeddings.to(device)

        num_queries = qs_padded.shape[0]

        # 모든 청크에 대한 스코어를 저장할 리스트
        all_chunk_scores = []

        # 각 사전 패딩된 청크에 대해 스코어 계산
        for chunk_idx, ps_padded in enumerate(self.prepadded_chunks):
            # ps_padded: [chunk_size, seq_len, 128] - 이미 GPU에 있음!
            # qs_padded: [num_queries, query_len, 128]

            # MaxSim 연산: 기존 score_multi_vector와 100% 동일
            # einsum("bnd,csd->bcns", q, p).max(dim=3)[0].sum(dim=2)
            #   b = num_queries, n = query_tokens, d = dim (128)
            #   c = chunk_images, s = image_tokens
            #   결과: [b, c, n, s] → max(dim=3) → [b, c, n] → sum(dim=2) → [b, c]

            scores = torch.einsum(
                "bnd,csd->bcns", qs_padded, ps_padded
            ).max(dim=3)[0].sum(dim=2)  # [num_queries, chunk_size]

            all_chunk_scores.append(scores)

        # 모든 청크 스코어를 연결 [num_queries, total_images]
        final_scores = torch.cat(all_chunk_scores, dim=1)

        return final_scores.to(torch.float32)

    def batch_search(self, queries: List[str]) -> List[List[str]]:
        """
        배치 검색 (GPU 적극 활용 버전)

        최적화:
        1. 사전 패딩된 청크 사용 → 매번 패딩 불필요
        2. GPU에서 전체 스코어 계산
        3. 초대형 배치로 GPU 활용률 극대화

        Args:
            queries: 검색 쿼리 리스트

        Returns:
            각 쿼리에 대한 검색 결과 리스트 (기존과 100% 동일)
        """
        if not queries:
            return []

        start_time = time.time()

        # 결과 저장용 (원본 순서 유지)
        results: List[Optional[List[str]]] = [None] * len(queries)

        # 캐시 조회
        uncached_indices = []
        uncached_queries = []

        if self.enable_cache and self.query_cache:
            for i, query in enumerate(queries):
                cached_result = self.query_cache.get(query)
                if cached_result is not None:
                    results[i] = cached_result
                else:
                    uncached_indices.append(i)
                    uncached_queries.append(query)
        else:
            uncached_indices = list(range(len(queries)))
            uncached_queries = queries

        # 캐시 미스된 쿼리만 처리
        if uncached_queries:
            device = self.vector_embed_model.embed_model.device

            # 쿼리 임베딩 계산
            batch_queries = self.vector_embed_model.processor.process_queries(
                uncached_queries
            ).to(device)

            with torch.no_grad():
                query_embeddings = self.vector_embed_model.embed_model(**batch_queries)

                # GPU 적극 활용: 사전 패딩된 청크로 스코어 계산
                scores = self._score_prepadded_chunks(query_embeddings, device)

            # Top-K 추출 (GPU에서 수행)
            values, indices = torch.topk(
                scores, k=min(self.image_nums, 10), dim=1
            )

            # 결과 매핑 및 캐시 저장
            indices_cpu = indices.cpu().tolist()
            for idx, (orig_idx, query) in enumerate(zip(uncached_indices, uncached_queries)):
                result = [self.nodes[i].metadata['file_name'] for i in indices_cpu[idx]]
                results[orig_idx] = result

                if self.enable_cache and self.query_cache:
                    self.query_cache.put(query, result)

        # 메모리 정리
        self._maybe_cleanup_memory()

        # 통계 업데이트
        elapsed = time.time() - start_time
        self._total_queries += len(queries)
        self._total_time += elapsed

        return results  # type: ignore

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 및 성능 통계 반환"""
        stats = {}

        if self.query_cache:
            stats.update(self.query_cache.get_stats())
        else:
            stats['cache_enabled'] = False

        # 성능 통계
        stats['total_queries'] = self._total_queries
        stats['total_time'] = f"{self._total_time:.2f}s"
        if self._total_queries > 0 and self._total_time > 0:
            stats['avg_queries_per_sec'] = f"{self._total_queries / self._total_time:.1f}"

        # GPU 메모리 상태
        mem_used = torch.cuda.memory_allocated(self.gpu_id) / 1024**3
        stats['gpu_memory'] = f"{mem_used:.2f}GB"

        # 설정 정보
        stats['config'] = {
            'prepad_chunk_size': GPUConfig.PREPAD_CHUNK_SIZE,
            'num_chunks': len(self.prepadded_chunks) if hasattr(self, 'prepadded_chunks') else 0,
            'gpu_id': self.gpu_id,
            'image_count': self.image_nums,
        }

        return stats

    def clear_cache(self):
        """캐시 초기화"""
        if self.query_cache:
            self.query_cache.clear()
            print("[SearchEngine] 쿼리 캐시 초기화 완료")

    def force_memory_cleanup(self):
        """강제 메모리 정리"""
        gc.collect()
        torch.cuda.empty_cache()
        print("[SearchEngine] 강제 메모리 정리 완료")

        # GPU 메모리 상태 출력
        mem_used = torch.cuda.memory_allocated(self.gpu_id) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(self.gpu_id) / 1024**3
        print(f"  GPU {self.gpu_id}: 사용={mem_used:.2f}GB, 예약={mem_reserved:.2f}GB")


if __name__ == '__main__':
    # 테스트
    print("=" * 60)
    print("SearchEngine GPU 적극 활용 버전 v2 테스트")
    print("=" * 60)

    search_engine = SearchEngine(
        dataset_dir='search_engine/corpus',
        embed_model_name='vidore/colqwen2-v1.0',
        gpu_id=2,
        chunk_size=16384,  # 16K 이미지씩 사전 패딩
    )

    # 단일 쿼리 테스트
    print("\n=== 단일 쿼리 테스트 ===")
    result = search_engine.batch_search(['travel market in APAC by 2020'])
    print(f"결과: {result}")

    # 배치 쿼리 테스트
    print("\n=== 배치 쿼리 테스트 (100개) ===")
    queries = [f"test query {i}" for i in range(100)]
    start = time.time()
    results = search_engine.batch_search(queries)
    elapsed = time.time() - start
    print(f"100개 쿼리 처리 시간: {elapsed:.2f}초 ({100/elapsed:.1f} queries/sec)")

    # 대용량 배치 테스트
    print("\n=== 대용량 배치 테스트 (1000개) ===")
    queries = [f"large batch test query {i}" for i in range(1000)]
    start = time.time()
    results = search_engine.batch_search(queries)
    elapsed = time.time() - start
    print(f"1000개 쿼리 처리 시간: {elapsed:.2f}초 ({1000/elapsed:.1f} queries/sec)")

    # 통계 출력
    print("\n=== 통계 ===")
    import json
    print(json.dumps(search_engine.get_cache_stats(), indent=2, ensure_ascii=False))
