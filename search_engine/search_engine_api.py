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

# 전역 변수: SearchEngine 인스턴스
search_engine = None


@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 SearchEngine 초기화"""
    global search_engine
    print("Initializing SearchEngine (GPU 적극 활용 버전)...")
    search_engine = SearchEngine(
        dataset_dir,
        embed_model_name='vidore/colqwen2-v1.0',
        gpu_id=2,              # GPU 2 사용
        chunk_size=16384,      # 16K 이미지씩 사전 패딩
    )
    print(f"SearchEngine initialized with {search_engine.image_nums:,} images")


@app.get(
    "/search",
    summary="Perform a search query.",
    description="GPU 최적화된 검색 엔진을 사용하여 검색을 수행합니다.",
    response_model=List[List[Dict[str, Any]]]
)
async def search(queries: List[str] = Query(...)):
    """
    검색 수행.

    Args:
        queries: 검색 쿼리 리스트

    Returns:
        각 쿼리에 대한 검색 결과 리스트
    """
    results_batch = search_engine.batch_search(queries)
    results_batch = [
        [
            dict(idx=idx, image_file=os.path.join('./search_engine/corpus/img', file))
            for idx, file in enumerate(query_results)
        ]
        for query_results in results_batch
    ]
    return results_batch


# =========================================================================
# 캐시 및 메모리 관리 엔드포인트
# =========================================================================
@app.get(
    "/cache/stats",
    summary="Get cache and performance statistics",
    description="캐시 히트율, GPU 메모리 사용량, 성능 통계를 반환합니다.",
)
async def cache_stats():
    """캐시 및 성능 통계 조회"""
    return search_engine.get_cache_stats()


@app.post(
    "/cache/clear",
    summary="Clear cache",
    description="쿼리 결과 캐시를 초기화합니다.",
)
async def cache_clear():
    """캐시 초기화"""
    search_engine.clear_cache()
    return {"status": "success", "message": "Cache cleared"}


@app.post(
    "/memory/cleanup",
    summary="Force memory cleanup",
    description="GPU 메모리 캐시를 강제로 정리합니다.",
)
async def memory_cleanup():
    """강제 메모리 정리"""
    search_engine.force_memory_cleanup()
    return {"status": "success", "message": "Memory cleanup completed"}


@app.get(
    "/health",
    summary="Health check",
    description="서버 상태 확인.",
)
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "image_count": search_engine.image_nums,
        "cache_stats": search_engine.get_cache_stats(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
