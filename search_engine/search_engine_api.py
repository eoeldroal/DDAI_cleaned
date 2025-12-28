"""
SearchEngine API - GPU 최적화 버전

최적화된 SearchEngine을 사용하는 FastAPI 서버.
기존 API 인터페이스와 100% 호환됩니다.
"""
from fastapi import FastAPI, Query
import uvicorn
from typing import List, Dict, Any
from search_engine import SearchEngine
from tqdm import tqdm
import os

dataset_dir = './search_engine/corpus'

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(
    title="Hybrid Search Engine API (GPU 적극 활용 버전)",
    description="GPU 적극 활용 검색 엔진 API. 사전 패딩된 청크 텐서로 성능 극대화.",
    version="2.1.0",
)

# 전역 변수: SearchEngine 인스턴스 리스트 및 로드 밸런서
search_engines = []
engine_cycle = None
# 사용 가능한 GPU ID 목록 (H200 8대)
GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]

from itertools import cycle
from fastapi.concurrency import run_in_threadpool

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 모든 GPU에 SearchEngine 초기화"""
    global search_engines, engine_cycle
    print(f"Initializing SearchEngines on GPUs {GPU_IDS} (Total {len(GPU_IDS)} GPUs)...")
    
    search_engines = []
    for gpu_id in GPU_IDS:
        try:
            print(f"Initializing engine on GPU {gpu_id}...")
            engine = SearchEngine(
                dataset_dir,
                embed_model_name='vidore/colqwen2-v1.0',
                gpu_id=gpu_id,
                chunk_size=16384,      # 16K 이미지씩 사전 패딩
            )
            search_engines.append(engine)
        except Exception as e:
            print(f"Failed to initialize engine on GPU {gpu_id}: {e}")
    
    if not search_engines:
        raise RuntimeError("No SearchEngine could be initialized.")
        
    engine_cycle = cycle(search_engines)
    print(f"Successfully initialized {len(search_engines)} SearchEngines.")


@app.get(
    "/search",
    summary="Perform a search query.",
    description="GPU 최적화된 검색 엔진을 사용하여 검색을 수행합니다.",
    response_model=List[List[Dict[str, Any]]]
)
async def search(queries: List[str] = Query(...)):
    """
    검색 수행.
    
    Multi-GPU Load Balancing을 적용하여 쿼리를 분산 처리합니다.
    """
    if not search_engines or engine_cycle is None:
        raise RuntimeError("Search engines not initialized")
        
    # Round-Robin 방식으로 엔진 선택
    engine = next(engine_cycle)
    
    # 별도 스레드에서 실행하여 이벤트 루프 블로킹 방지 + 병렬 처리
    results_batch = await run_in_threadpool(engine.batch_search, queries)
    
    results_batch = [
        [
            dict(idx=idx, image_file=os.path.join('./search_engine/corpus/img', file))
            for idx, file in enumerate(query_results)
        ]
        for query_results in results_batch
    ]
    return results_batch


# =========================================================================
# 캐시 및 메모리 관리 엔드포인트 (멀티 GPU 대응)
# =========================================================================
@app.get(
    "/cache/stats",
    summary="Get cache and performance statistics",
    description="모든 GPU 엔진의 통계를 반환합니다.",
)
async def cache_stats():
    """캐시 및 성능 통계 조회 (전체 합계 및 개별)"""
    if not search_engines:
        return {"error": "No engines"}
    
    stats = {
        "total_engines": len(search_engines),
        "engines": []
    }
    
    for i, engine in enumerate(search_engines):
        s = engine.get_cache_stats()
        s['gpu_id'] = engine.gpu_id
        stats['engines'].append(s)
        
    return stats


@app.post(
    "/cache/clear",
    summary="Clear cache",
    description="모든 엔진의 캐시를 초기화합니다.",
)
async def cache_clear():
    """캐시 초기화"""
    for engine in search_engines:
        engine.clear_cache()
    return {"status": "success", "message": "All caches cleared"}


@app.post(
    "/memory/cleanup",
    summary="Force memory cleanup",
    description="모든 GPU의 메모리를 정리합니다.",
)
async def memory_cleanup():
    """강제 메모리 정리"""
    for engine in search_engines:
        engine.force_memory_cleanup()
    return {"status": "success", "message": "All memory cleanups completed"}


@app.get(
    "/health",
    summary="Health check",
    description="서버 상태 확인.",
)
async def health_check():
    """서버 상태 확인"""
    if not search_engines:
        return {"status": "unhealthy", "message": "No engines initialized"}
        
    return {
        "status": "healthy",
        "active_gpus": len(search_engines),
        "total_images": sum(e.image_nums for e in search_engines) # Note: 중복됨
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
