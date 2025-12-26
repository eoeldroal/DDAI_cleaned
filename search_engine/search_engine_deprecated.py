import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Mapping, Any, Dict
import json
from tqdm import tqdm
import torch
import hashlib
from collections import OrderedDict
import threading

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo, ImageNode
from vl_embedding import VL_Embedding


# =========================================================================
# [NEW] 쿼리 결과 LRU 캐시
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
    def __init__(
        self,
        dataset_dir='search_engine/corpus',
        node_dir_prefix='colqwen_ingestion',
        embed_model_name='vidore/colqwen2-v1.0',
        cache_size: int = 10000,  # [NEW] 캐시 크기
        enable_cache: bool = True  # [NEW] 캐시 활성화 여부
    ):
        self.workers = 1
        self.dataset_dir = dataset_dir
        self.node_dir = os.path.join(self.dataset_dir, node_dir_prefix)
        self.vector_embed_model = VL_Embedding(model=embed_model_name, mode='image')

        # [NEW] 쿼리 결과 캐시 초기화
        self.enable_cache = enable_cache
        self.query_cache = QueryResultCache(max_size=cache_size) if enable_cache else None

        self.query_engine = self.load_query_engine()

        if enable_cache:
            print(f"[SearchEngine] 쿼리 캐시 활성화 (max_size={cache_size})")

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
            def parse_file(file,node_dir):
                input_file = os.path.join(node_dir, file)
                suffix = input_file.split('.')[-1]
                if suffix != 'node':
                    return []
                return nodefile2node(input_file)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # results = list(tqdm(executor.map(parse_file, files, self.node_dir), total=len(files)))
                results = list(tqdm(executor.map(parse_file, files, [self.node_dir]*len(files)), total=len(files)))
            # 合并所有线程的结果
            for result in results:
                parsed_files.extend(result)
        return parsed_files
        
    def load_query_engine(self):
        print('Loading nodes...')
        self.nodes = self.load_nodes()
        self.embedding_img = [torch.tensor(node.embedding).view(-1, 128).bfloat16() for node in tqdm(self.nodes, desc="Creating Embeddings")]
        self.embedding_img = [tensor.to(self.vector_embed_model.embed_model.device) for tensor in tqdm(self.embedding_img, desc="Moving to Device")]
        self.image_nums = len(self.embedding_img)

    def load_node_postprocessors(self):
        return []

    def batch_search(self, queries: List[str]) -> List[List[str]]:
        """
        배치 검색 (캐시 지원)

        캐시가 활성화된 경우:
        1. 캐시 히트된 쿼리는 바로 결과 반환
        2. 캐시 미스된 쿼리만 임베딩 계산
        3. 새 결과는 캐시에 저장
        """
        if not queries:
            return []

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
            # 캐시 비활성화: 모든 쿼리 처리
            uncached_indices = list(range(len(queries)))
            uncached_queries = queries

        # 캐시 미스된 쿼리만 임베딩 계산
        if uncached_queries:
            batch_queries = self.vector_embed_model.processor.process_queries(
                uncached_queries
            ).to(self.vector_embed_model.embed_model.device)

            with torch.no_grad():
                query_embeddings = self.vector_embed_model.embed_model(**batch_queries)

            scores = self.vector_embed_model.processor.score_multi_vector(
                query_embeddings,
                self.embedding_img,
                batch_size=256,
                device=self.vector_embed_model.embed_model.device
            )

            values, indices = torch.topk(scores, k=min(self.image_nums, 10), dim=1)

            # 결과 매핑 및 캐시 저장
            for idx, (orig_idx, query) in enumerate(zip(uncached_indices, uncached_queries)):
                result = [self.nodes[i].metadata['file_name'] for i in indices[idx]]
                results[orig_idx] = result

                # 캐시에 저장
                if self.enable_cache and self.query_cache:
                    self.query_cache.put(query, result)

        return results  # type: ignore

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        if self.query_cache:
            return self.query_cache.get_stats()
        return {'enabled': False}

    def clear_cache(self):
        """캐시 초기화"""
        if self.query_cache:
            self.query_cache.clear()
            print("[SearchEngine] 캐시 초기화 완료")


if __name__ == '__main__':
    search_engine = SearchEngine(dataset_dir='search_engine/corpus',embed_model_name='vidore/colqwen2-v1.0')
    print(search_engine.batch_search(['travel market in APAC by 2020']))
    

    