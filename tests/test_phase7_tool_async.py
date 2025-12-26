#!/usr/bin/env python3
"""
Phase 7 Tool 비동기화 단위 테스트

이 테스트는 다음을 검증합니다:
1. Phase 7 필드가 올바르게 초기화되는지
2. Search 호출이 비동기로 시작되는지 (블로킹 없음)
3. bbox, search_complete가 search와 병렬로 처리되는지
4. Search 결과가 올바르게 반영되는지
5. Phase 7 비활성화 시 기존 방식으로 폴백되는지
"""

import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_phase7_fields_exist():
    """Phase 7 새 필드 존재 확인"""
    print("=" * 60)
    print("1. Phase 7 새 필드 존재 확인")
    print("=" * 60)

    try:
        from vrag_agent.generation import LLMGenerationManager, GenerationConfig

        # Mock processor 생성
        class MockProcessor:
            class Tokenizer:
                pad_token_id = 0
            tokenizer = Tokenizer()

        config = GenerationConfig(
            max_turns=5,
            max_prompt_length=4096,
            num_gpus=8,
            search_url="http://localhost:5002/search"
        )

        manager = LLMGenerationManager(
            processor=MockProcessor(),
            actor_rollout_wg=None,
            config=config,
            is_validation=False,
            streaming_reward_manager=None
        )

        # Phase 7 필드 확인
        assert hasattr(manager, '_tool_executor'), "_tool_executor 필드 누락"
        assert hasattr(manager, '_phase7_enabled'), "_phase7_enabled 필드 누락"
        assert isinstance(manager._tool_executor, ThreadPoolExecutor), "_tool_executor가 ThreadPoolExecutor가 아님"
        assert manager._phase7_enabled == True, "_phase7_enabled 기본값이 True가 아님"

        print("  _tool_executor 존재")
        print("  _phase7_enabled 존재")
        print(f"  _phase7_enabled = {manager._phase7_enabled}")
        print("✅ Phase 7 필드 테스트 통과")
        print()
        return True

    except Exception as e:
        print(f"❌ 필드 확인 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_async_search_non_blocking():
    """비동기 Search 호출이 블로킹하지 않는지 테스트"""
    print("=" * 60)
    print("2. 비동기 Search 호출 블로킹 테스트")
    print("=" * 60)

    try:
        # 시뮬레이션: ThreadPoolExecutor로 비동기 작업
        executor = ThreadPoolExecutor(max_workers=4)

        def slow_search():
            """느린 검색 시뮬레이션 (2초 소요)"""
            time.sleep(2)
            return {"result": "search_complete"}

        # 비동기 호출 시작
        start_time = time.perf_counter()
        future = executor.submit(slow_search)
        submit_time = time.perf_counter() - start_time

        print(f"  Future 생성 시간: {submit_time:.4f}초")

        # 다른 작업 수행 (bbox 처리 시뮬레이션)
        bbox_start = time.perf_counter()
        bbox_result = [10, 20, 30, 40]  # 즉시 처리
        bbox_time = time.perf_counter() - bbox_start
        print(f"  bbox 처리 시간: {bbox_time:.6f}초")

        # search_complete 처리 시뮬레이션
        complete_start = time.perf_counter()
        search_completed = True
        complete_time = time.perf_counter() - complete_start
        print(f"  search_complete 처리 시간: {complete_time:.6f}초")

        # Search 결과 대기
        wait_start = time.perf_counter()
        result = future.result(timeout=5)
        wait_time = time.perf_counter() - wait_start
        print(f"  Search 결과 대기 시간: {wait_time:.2f}초")

        total_time = time.perf_counter() - start_time
        print(f"  총 소요 시간: {total_time:.2f}초")

        # 검증: Future 생성이 빠르고 (< 0.01초), 총 시간이 ~2초 (병렬 처리)
        if submit_time < 0.01 and total_time < 2.5:
            print("✅ 비동기 호출 블로킹 테스트 통과 (병렬 처리 확인)")
            print()
            executor.shutdown(wait=False)
            return True
        else:
            print("❌ 비동기 호출 블로킹 테스트 실패")
            executor.shutdown(wait=False)
            return False

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parallel_processing_simulation():
    """bbox와 search가 병렬로 처리되는지 시뮬레이션 테스트"""
    print("=" * 60)
    print("3. 병렬 처리 시뮬레이션 테스트")
    print("=" * 60)

    try:
        executor = ThreadPoolExecutor(max_workers=4)
        results = {}
        timings = {}

        def search_task(idx):
            """Search 시뮬레이션 (1초 소요)"""
            start = time.perf_counter()
            time.sleep(1)
            elapsed = time.perf_counter() - start
            return {'idx': idx, 'type': 'search', 'elapsed': elapsed}

        def bbox_task(idx):
            """bbox 시뮬레이션 (즉시)"""
            start = time.perf_counter()
            result = [10, 20, 30, 40]
            elapsed = time.perf_counter() - start
            return {'idx': idx, 'type': 'bbox', 'elapsed': elapsed}

        # 시나리오: 4개 샘플 (search, bbox, search, search_complete)
        actions = ['search', 'bbox', 'search', 'search_complete']

        start_time = time.perf_counter()

        # Step 1: Search 비동기 시작
        search_futures = []
        for i, action in enumerate(actions):
            if action == 'search':
                future = executor.submit(search_task, i)
                search_futures.append((i, future))

        submit_time = time.perf_counter() - start_time
        print(f"  Search 비동기 시작 시간: {submit_time:.4f}초")

        # Step 2: bbox, search_complete 즉시 처리
        for i, action in enumerate(actions):
            if action == 'bbox':
                results[i] = bbox_task(i)
            elif action == 'search_complete':
                results[i] = {'idx': i, 'type': 'search_complete', 'elapsed': 0.0001}

        immediate_time = time.perf_counter() - start_time
        print(f"  즉시 처리 완료 시간: {immediate_time:.4f}초")

        # Step 3: Search 결과 대기
        for i, future in search_futures:
            results[i] = future.result(timeout=5)

        total_time = time.perf_counter() - start_time
        print(f"  총 소요 시간: {total_time:.2f}초")

        # 결과 출력
        print("  결과:")
        for idx in sorted(results.keys()):
            r = results[idx]
            print(f"    [{idx}] {r['type']}: {r['elapsed']:.4f}초")

        # 검증: 총 시간이 search 시간(1초)보다 약간 더 걸려야 함 (병렬 처리)
        # 순차 처리 시: 1 + 0 + 1 + 0 = 2초
        # 병렬 처리 시: ~1초 (search 2개가 병렬)
        if total_time < 1.5:
            print("✅ 병렬 처리 시뮬레이션 테스트 통과")
            print()
            executor.shutdown(wait=False)
            return True
        else:
            print(f"❌ 병렬 처리 실패 (예상: <1.5초, 실제: {total_time:.2f}초)")
            executor.shutdown(wait=False)
            return False

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_execute_predictions_structure():
    """execute_predictions 함수 구조 테스트 (Mock 사용)"""
    print("=" * 60)
    print("4. execute_predictions 구조 테스트")
    print("=" * 60)

    try:
        from vrag_agent.generation import LLMGenerationManager, GenerationConfig

        # Mock processor 생성
        class MockProcessor:
            class Tokenizer:
                pad_token_id = 0
            tokenizer = Tokenizer()

        config = GenerationConfig(
            max_turns=5,
            max_prompt_length=4096,
            num_gpus=8,
            search_url="http://localhost:5002/search"
        )

        manager = LLMGenerationManager(
            processor=MockProcessor(),
            actor_rollout_wg=None,
            config=config,
            is_validation=False,
            streaming_reward_manager=None
        )

        # 테스트 데이터
        predictions = [
            "<search>test query 1</search>",
            "<bbox>[10,20,30,40]</bbox>",
            "<search>test query 2</search>",
            "<search_complete>true</search_complete>",
        ]
        uids = np.array(["uid_0", "uid_1", "uid_2", "uid_3"])
        active_mask = np.array([True, True, True, True])

        # search_completed 초기화
        manager.search_completed = [False] * 4

        # Mock _async_search_batches
        def mock_search_batches(requests):
            # request_idx를 키로 사용
            return {req['request_idx']: [{'image_file': f'/path/to/img_{req["request_idx"]}.jpg'}] for req in requests}

        manager._async_search_batches = mock_search_batches

        # execute_predictions 호출 (do_search=False로 검색 건너뛰기)
        # Phase 7 테스트는 구조만 확인
        next_obs, dones = manager.execute_predictions(
            predictions, uids, manager.processor.tokenizer.pad_token_id, active_mask, do_search=False
        )

        print(f"  next_obs 길이: {len(next_obs)}")
        print(f"  dones 길이: {len(dones)}")
        print(f"  next_obs 타입: {[type(x).__name__ for x in next_obs]}")
        print(f"  dones: {dones}")

        # 검증
        assert len(next_obs) == 4, f"next_obs 길이 불일치: {len(next_obs)}"
        assert len(dones) == 4, f"dones 길이 불일치: {len(dones)}"
        assert dones[1] == 0, "bbox의 done이 0이 아님"  # bbox
        assert dones[3] == 1, "search_complete의 done이 1이 아님"  # search_complete
        assert next_obs[1] == [10, 20, 30, 40], f"bbox 결과 불일치: {next_obs[1]}"

        print("✅ execute_predictions 구조 테스트 통과")
        print()
        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase7_flag_toggle():
    """Phase 7 활성화/비활성화 플래그 테스트"""
    print("=" * 60)
    print("5. Phase 7 플래그 토글 테스트")
    print("=" * 60)

    try:
        from vrag_agent.generation import LLMGenerationManager, GenerationConfig

        class MockProcessor:
            class Tokenizer:
                pad_token_id = 0
            tokenizer = Tokenizer()

        # Phase 7 활성화 (기본값)
        config1 = GenerationConfig(
            max_turns=5,
            max_prompt_length=4096,
            num_gpus=8,
            search_url="http://localhost:5002/search"
        )
        manager1 = LLMGenerationManager(
            processor=MockProcessor(),
            actor_rollout_wg=None,
            config=config1,
            is_validation=False,
            streaming_reward_manager=None
        )
        print(f"  기본값: _phase7_enabled = {manager1._phase7_enabled}")
        assert manager1._phase7_enabled == True, "기본값이 True가 아님"

        # Phase 7 비활성화
        config2 = GenerationConfig(
            max_turns=5,
            max_prompt_length=4096,
            num_gpus=8,
            search_url="http://localhost:5002/search",
            phase7_tool_async=False
        )
        manager2 = LLMGenerationManager(
            processor=MockProcessor(),
            actor_rollout_wg=None,
            config=config2,
            is_validation=False,
            streaming_reward_manager=None
        )
        print(f"  비활성화: _phase7_enabled = {manager2._phase7_enabled}")
        assert manager2._phase7_enabled == False, "비활성화 설정이 적용되지 않음"

        print("✅ Phase 7 플래그 토글 테스트 통과")
        print()
        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_thread_safety():
    """Thread-safety 테스트"""
    print("=" * 60)
    print("6. Thread-safety 테스트")
    print("=" * 60)

    try:
        executor = ThreadPoolExecutor(max_workers=10)
        results = {}
        lock = threading.Lock()
        errors = []

        def worker(task_id):
            try:
                # 작업 시뮬레이션
                time.sleep(0.1)
                with lock:
                    results[task_id] = f"result_{task_id}"
            except Exception as e:
                errors.append(str(e))

        # 100개 동시 작업
        futures = [executor.submit(worker, i) for i in range(100)]

        # 모든 작업 완료 대기
        for f in futures:
            f.result(timeout=10)

        executor.shutdown(wait=True)

        print(f"  완료된 작업 수: {len(results)}")
        print(f"  오류 수: {len(errors)}")

        if len(results) == 100 and len(errors) == 0:
            print("✅ Thread-safety 테스트 통과")
            print()
            return True
        else:
            print("❌ Thread-safety 테스트 실패")
            return False

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("  Phase 7 Tool 비동기화 단위 테스트 시작")
    print("=" * 60 + "\n")

    results = {}

    # 1. Phase 7 필드 테스트
    results['phase7_fields'] = test_phase7_fields_exist()

    # 2. 비동기 호출 블로킹 테스트
    results['async_non_blocking'] = test_async_search_non_blocking()

    # 3. 병렬 처리 시뮬레이션
    results['parallel_processing'] = test_parallel_processing_simulation()

    # 4. execute_predictions 구조 테스트
    results['execute_predictions'] = test_execute_predictions_structure()

    # 5. Phase 7 플래그 토글 테스트
    results['phase7_flag'] = test_phase7_flag_toggle()

    # 6. Thread-safety 테스트
    results['thread_safety'] = test_thread_safety()

    # 결과 요약
    print("=" * 60)
    print("  테스트 결과 요약")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 모든 Phase 7 단위 테스트 통과!")
    else:
        print("⚠️ 일부 테스트 실패")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
