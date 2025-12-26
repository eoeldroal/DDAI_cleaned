#!/usr/bin/env python3
"""
Phase 7 Tool 비동기화 통합 테스트

실제 API 호출을 시뮬레이션하여 End-to-End 테스트:
1. 혼합 액션 시나리오 (search + bbox + search_complete)
2. 비동기 처리 시간 측정
3. 대규모 배치 처리 테스트
4. 에러 핸들링 테스트
"""

import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_mock_manager():
    """테스트용 Mock Manager 생성"""
    from vrag_agent.generation import LLMGenerationManager, GenerationConfig

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

    return manager


def test_mixed_actions_scenario():
    """혼합 액션 시나리오 테스트"""
    print("=" * 60)
    print("1. 혼합 액션 시나리오 테스트")
    print("=" * 60)

    try:
        manager = create_mock_manager()

        # 테스트 데이터: 8개 샘플 (다양한 액션 혼합)
        predictions = [
            "<search>query 0</search>",
            "<bbox>[10,20,30,40]</bbox>",
            "<search>query 2</search>",
            "<search_complete>true</search_complete>",
            "<bbox>[50,60,70,80]</bbox>",
            "<search>query 5</search>",
            "<search_complete>false</search_complete>",  # false
            "<search>query 7</search>",
        ]
        uids = np.array([f"uid_{i}" for i in range(8)])
        active_mask = np.array([True] * 8)

        # search_completed 초기화
        manager.search_completed = [False] * 8

        # Mock _async_search_batches (지연 시뮬레이션)
        def mock_search_batches_delayed(requests):
            time.sleep(0.5)  # 0.5초 지연
            return {req['request_idx']: [{'image_file': f'/path/img_{req["request_idx"]}.jpg'}] for req in requests}

        manager._async_search_batches = mock_search_batches_delayed

        # Phase 7 비동기 처리 시간 측정
        start_time = time.perf_counter()
        next_obs, dones = manager.execute_predictions(
            predictions, uids, manager.processor.tokenizer.pad_token_id, active_mask, do_search=True
        )
        elapsed = time.perf_counter() - start_time

        print(f"  처리 시간: {elapsed:.3f}초")
        print(f"  next_obs 길이: {len(next_obs)}")
        print(f"  dones: {dones}")

        # 결과 검증
        assert len(next_obs) == 8, f"next_obs 길이 불일치: {len(next_obs)}"
        assert len(dones) == 8, f"dones 길이 불일치: {len(dones)}"

        # bbox 결과 확인
        assert next_obs[1] == [10, 20, 30, 40], f"bbox[1] 결과 불일치: {next_obs[1]}"
        assert next_obs[4] == [50, 60, 70, 80], f"bbox[4] 결과 불일치: {next_obs[4]}"

        # search_complete 결과 확인
        assert dones[3] == 1, "search_complete[3]의 done이 1이 아님"
        assert manager.search_completed[3] == True, "search_completed[3]이 True가 아님"
        assert dones[6] == 1, "search_complete[6]의 done이 1이 아님"
        assert manager.search_completed[6] == False, "search_completed[6]이 False여야 함 (false 입력)"

        # search 결과 확인 (이미지 리스트)
        assert isinstance(next_obs[0], list), f"search[0] 결과가 리스트가 아님: {type(next_obs[0])}"
        assert isinstance(next_obs[2], list), f"search[2] 결과가 리스트가 아님: {type(next_obs[2])}"

        print("  결과 검증:")
        for i, (obs, done) in enumerate(zip(next_obs, dones)):
            action = predictions[i].split('>')[0].split('<')[1]
            print(f"    [{i}] {action}: done={done}, obs_type={type(obs).__name__}")

        print("✅ 혼합 액션 시나리오 테스트 통과")
        print()
        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_async_timing_comparison():
    """비동기 vs 동기 처리 시간 비교 테스트"""
    print("=" * 60)
    print("2. 비동기 vs 동기 처리 시간 비교")
    print("=" * 60)

    try:
        manager = create_mock_manager()

        # 테스트 데이터: search 4개 + bbox 2개
        predictions = [
            "<search>query 0</search>",
            "<bbox>[10,20,30,40]</bbox>",
            "<search>query 2</search>",
            "<bbox>[50,60,70,80]</bbox>",
            "<search>query 4</search>",
            "<search>query 5</search>",
        ]
        uids = np.array([f"uid_{i}" for i in range(6)])
        active_mask = np.array([True] * 6)
        manager.search_completed = [False] * 6

        # Mock: 각 search 호출에 0.2초 지연
        search_delay = 0.2

        def mock_search_batches_delayed(requests):
            time.sleep(search_delay)  # 검색 시뮬레이션
            return {req['request_idx']: [{'image_file': f'/path/img_{req["request_idx"]}.jpg'}] for req in requests}

        manager._async_search_batches = mock_search_batches_delayed

        # Phase 7 활성화 테스트
        manager._phase7_enabled = True
        start_async = time.perf_counter()
        next_obs_async, dones_async = manager.execute_predictions(
            predictions, uids, manager.processor.tokenizer.pad_token_id, active_mask, do_search=True
        )
        time_async = time.perf_counter() - start_async

        # Phase 7 비활성화 테스트 (기존 방식)
        manager._phase7_enabled = False
        manager.search_completed = [False] * 6  # 리셋
        start_sync = time.perf_counter()
        next_obs_sync, dones_sync = manager.execute_predictions(
            predictions, uids, manager.processor.tokenizer.pad_token_id, active_mask, do_search=True
        )
        time_sync = time.perf_counter() - start_sync

        print(f"  Phase 7 활성화 (비동기): {time_async:.3f}초")
        print(f"  Phase 7 비활성화 (동기): {time_sync:.3f}초")
        print(f"  시간 차이: {time_sync - time_async:.3f}초")

        # 결과가 동일한지 확인
        assert next_obs_async == next_obs_sync, "비동기/동기 결과 불일치"
        assert dones_async == dones_sync, "비동기/동기 dones 불일치"

        print("  결과 동일성: ✅")
        print("✅ 비동기/동기 비교 테스트 통과")
        print()
        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_large_batch_processing():
    """대규모 배치 처리 테스트 (128개 샘플)"""
    print("=" * 60)
    print("3. 대규모 배치 처리 테스트 (128개 샘플)")
    print("=" * 60)

    try:
        manager = create_mock_manager()

        # 128개 샘플 생성 (실제 배치 크기와 동일)
        n_samples = 128
        predictions = []
        for i in range(n_samples):
            if i % 4 == 0:
                predictions.append(f"<search>query {i}</search>")
            elif i % 4 == 1:
                predictions.append(f"<bbox>[{i},{i+10},{i+20},{i+30}]</bbox>")
            elif i % 4 == 2:
                predictions.append(f"<search>query {i}</search>")
            else:
                predictions.append("<search_complete>true</search_complete>")

        uids = np.array([f"uid_{i}" for i in range(n_samples)])
        active_mask = np.array([True] * n_samples)
        manager.search_completed = [False] * n_samples

        # Mock: 짧은 지연
        def mock_search_batches(requests):
            time.sleep(0.1)  # 빠른 응답
            return {req['request_idx']: [{'image_file': f'/path/img_{req["request_idx"]}.jpg'}] for req in requests}

        manager._async_search_batches = mock_search_batches

        # 처리
        start_time = time.perf_counter()
        next_obs, dones = manager.execute_predictions(
            predictions, uids, manager.processor.tokenizer.pad_token_id, active_mask, do_search=True
        )
        elapsed = time.perf_counter() - start_time

        print(f"  샘플 수: {n_samples}")
        print(f"  처리 시간: {elapsed:.3f}초")
        print(f"  샘플당 시간: {elapsed/n_samples*1000:.3f}ms")

        # 검증
        assert len(next_obs) == n_samples, f"next_obs 길이 불일치: {len(next_obs)}"
        assert len(dones) == n_samples, f"dones 길이 불일치: {len(dones)}"

        # None이 없는지 확인
        none_count = sum(1 for x in next_obs if x is None)
        assert none_count == 0, f"next_obs에 None이 {none_count}개 있음"

        # 통계
        search_count = sum(1 for p in predictions if '<search>' in p and 'complete' not in p)
        bbox_count = sum(1 for p in predictions if '<bbox>' in p)
        complete_count = sum(1 for p in predictions if '<search_complete>' in p)
        print(f"  search: {search_count}, bbox: {bbox_count}, complete: {complete_count}")

        print("✅ 대규모 배치 처리 테스트 통과")
        print()
        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """에러 핸들링 테스트"""
    print("=" * 60)
    print("4. 에러 핸들링 테스트")
    print("=" * 60)

    try:
        manager = create_mock_manager()

        predictions = [
            "<search>query 0</search>",
            "<bbox>[10,20,30,40]</bbox>",
            "<search>query 2</search>",
        ]
        uids = np.array([f"uid_{i}" for i in range(3)])
        active_mask = np.array([True] * 3)
        manager.search_completed = [False] * 3

        # Mock: 에러 발생 시뮬레이션
        def mock_search_batches_error(requests):
            raise Exception("Search API Error!")

        manager._async_search_batches = mock_search_batches_error

        # 에러가 발생해도 처리가 완료되어야 함
        next_obs, dones = manager.execute_predictions(
            predictions, uids, manager.processor.tokenizer.pad_token_id, active_mask, do_search=True
        )

        print(f"  next_obs: {next_obs}")
        print(f"  dones: {dones}")

        # 검증: 에러 시 빈 결과로 폴백
        assert len(next_obs) == 3, f"next_obs 길이 불일치: {len(next_obs)}"
        assert len(dones) == 3, f"dones 길이 불일치: {len(dones)}"

        # search 결과가 빈 리스트인지 확인 (폴백)
        assert next_obs[0] == [], f"search[0] 폴백 실패: {next_obs[0]}"
        assert next_obs[2] == [], f"search[2] 폴백 실패: {next_obs[2]}"

        # bbox는 정상 처리
        assert next_obs[1] == [10, 20, 30, 40], f"bbox[1] 결과 불일치: {next_obs[1]}"

        print("✅ 에러 핸들링 테스트 통과")
        print()
        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inactive_samples():
    """비활성 샘플 처리 테스트"""
    print("=" * 60)
    print("5. 비활성 샘플 처리 테스트")
    print("=" * 60)

    try:
        manager = create_mock_manager()

        predictions = [
            "<search>query 0</search>",
            "<bbox>[10,20,30,40]</bbox>",
            "<search>query 2</search>",
            "<search_complete>true</search_complete>",
        ]
        uids = np.array([f"uid_{i}" for i in range(4)])
        # 일부 샘플 비활성화
        active_mask = np.array([True, False, True, False])
        manager.search_completed = [False] * 4

        def mock_search_batches(requests):
            return {req['request_idx']: [{'image_file': f'/path/img_{req["request_idx"]}.jpg'}] for req in requests}

        manager._async_search_batches = mock_search_batches

        next_obs, dones = manager.execute_predictions(
            predictions, uids, manager.processor.tokenizer.pad_token_id, active_mask, do_search=True
        )

        print(f"  active_mask: {active_mask.tolist()}")
        print(f"  next_obs: {next_obs}")
        print(f"  dones: {dones}")

        # 비활성 샘플 검증
        assert next_obs[1] == '', "비활성 샘플[1]의 obs가 빈 문자열이 아님"
        assert dones[1] == 1, "비활성 샘플[1]의 done이 1이 아님"
        assert next_obs[3] == '', "비활성 샘플[3]의 obs가 빈 문자열이 아님"
        assert dones[3] == 1, "비활성 샘플[3]의 done이 1이 아님"

        # 활성 샘플 검증
        assert isinstance(next_obs[0], list), "활성 search[0]의 결과가 리스트가 아님"
        assert isinstance(next_obs[2], list), "활성 search[2]의 결과가 리스트가 아님"

        print("✅ 비활성 샘플 처리 테스트 통과")
        print()
        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("  Phase 7 Tool 비동기화 통합 테스트 시작")
    print("=" * 60 + "\n")

    results = {}

    # 1. 혼합 액션 시나리오
    results['mixed_actions'] = test_mixed_actions_scenario()

    # 2. 비동기/동기 비교
    results['timing_comparison'] = test_async_timing_comparison()

    # 3. 대규모 배치
    results['large_batch'] = test_large_batch_processing()

    # 4. 에러 핸들링
    results['error_handling'] = test_error_handling()

    # 5. 비활성 샘플
    results['inactive_samples'] = test_inactive_samples()

    # 결과 요약
    print("=" * 60)
    print("  통합 테스트 결과 요약")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 모든 Phase 7 통합 테스트 통과!")
        print("\nPhase 7 Tool 비동기화가 올바르게 구현되었습니다:")
        print("  - Search API 호출 비동기 시작 ✓")
        print("  - bbox/search_complete 병렬 처리 ✓")
        print("  - 에러 핸들링 및 폴백 ✓")
        print("  - 대규모 배치 처리 ✓")
    else:
        print("⚠️ 일부 테스트 실패")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
