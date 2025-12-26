#!/usr/bin/env python3
"""
Phase 6 통합 테스트

실제 API를 사용한 End-to-End 테스트:
1. Frozen Generator 배치 호출 테스트
2. 백그라운드 스레드에서 전체 파이프라인 테스트
3. 다중 프롬프트 동시 처리 테스트
"""

import os
import sys
import asyncio
import threading
import time
from typing import Dict, List

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# .env 로드
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))


def test_frozen_generator_batch_call():
    """Frozen Generator 배치 호출 통합 테스트"""
    print("=" * 60)
    print("1. Frozen Generator 배치 호출 통합 테스트")
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
            search_url="http://localhost:5002/search",
            frozen_model="qwen2.5-vl-72b-instruct",
            frozen_max_tokens=256,
            frozen_max_concurrent=10,
        )

        manager = LLMGenerationManager(
            processor=MockProcessor(),
            actor_rollout_wg=None,
            config=config,
            is_validation=False,
            streaming_reward_manager=None
        )

        # 테스트 데이터
        indices = [0, 1, 2]
        questions = [
            "What is 5 + 5?",
            "What is the capital of France?",
            "What is 2 * 3?",
        ]
        images_list = [[], [], []]  # 이미지 없이 텍스트만 테스트

        print("  호출 중...")
        start_time = time.perf_counter()

        # 배치 호출
        results = manager._call_frozen_generator_batch(indices, questions, images_list)

        elapsed = time.perf_counter() - start_time

        print(f"  소요 시간: {elapsed:.2f}초")
        print("  결과:")

        success_count = 0
        for idx, q in zip(indices, questions):
            answer = results.get(idx, "")
            status = "✅" if answer else "❌"
            if answer:
                success_count += 1
            print(f"    {status} [{idx}] {q}")
            print(f"        → {answer[:100]}..." if len(answer) > 100 else f"        → {answer}")

        print(f"\n  성공률: {success_count}/{len(indices)}")

        if success_count == len(indices):
            print("✅ Frozen Generator 배치 호출 테스트 통과")
            print()
            return True
        else:
            print("❌ 일부 실패")
            return False

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_background_pipeline_simulation():
    """백그라운드 파이프라인 시뮬레이션 테스트"""
    print("=" * 60)
    print("2. 백그라운드 파이프라인 시뮬레이션 테스트")
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
            search_url="http://localhost:5002/search",
            frozen_model="qwen2.5-vl-72b-instruct",
            frozen_max_tokens=128,
            frozen_max_concurrent=20,
        )

        manager = LLMGenerationManager(
            processor=MockProcessor(),
            actor_rollout_wg=None,
            config=config,
            is_validation=False,
            streaming_reward_manager=None
        )

        # 시뮬레이션 데이터 설정
        manager.questions = ["What is 100 + 200?", "What is 300 + 400?"]
        manager.retrievaled_images = [[], []]
        manager.cropped_images = [[], []]
        manager.generated_answers = {}
        manager._streaming_frozen_generated = set()
        manager._pending_threads = []

        # Mock streaming reward manager
        class MockStreamingRewardManager:
            def __init__(self):
                self.submissions = []

            def submit_prompt(self, uid, sample_indices, samples_data):
                self.submissions.append({
                    'uid': uid,
                    'sample_indices': sample_indices,
                    'samples_data': samples_data
                })
                print(f"  [Mock] submit_prompt called: uid={uid}, indices={sample_indices}")

        mock_rm = MockStreamingRewardManager()
        manager.streaming_reward_manager = mock_rm

        # 백그라운드 처리 시뮬레이션
        indices = [0, 1]
        prompt_id = "test_prompt_001"
        status = {'completed_samples': 2, 'total_samples': 2, 'submitted': False}

        print("  백그라운드 스레드 시작...")
        start_time = time.perf_counter()

        # 백그라운드 스레드로 처리
        thread = threading.Thread(
            target=manager._process_prompt_background,
            args=(indices, prompt_id, status),
            daemon=True
        )
        thread.start()

        spawn_time = time.perf_counter() - start_time
        print(f"  스레드 시작 시간: {spawn_time:.4f}초 (블로킹 없음)")

        # 메인 스레드 작업 시뮬레이션
        print("  메인 스레드: 다른 작업 수행 중... (시뮬레이션)")
        time.sleep(0.1)
        print("  메인 스레드: 작업 완료")

        # 백그라운드 스레드 완료 대기
        thread.join(timeout=60)

        total_time = time.perf_counter() - start_time

        # 결과 검증
        print("\n  결과 검증:")
        print(f"    generated_answers: {manager.generated_answers}")
        print(f"    _streaming_frozen_generated: {manager._streaming_frozen_generated}")
        print(f"    submit_prompt 호출 횟수: {len(mock_rm.submissions)}")

        if mock_rm.submissions:
            submission = mock_rm.submissions[0]
            print(f"    제출된 samples_data:")
            for i, data in enumerate(submission['samples_data']):
                print(f"      [{i}] query: {data['query']}")
                print(f"          generated_answer: {data.get('generated_answer', 'N/A')[:50]}...")

        # 검증 조건
        success = (
            len(manager.generated_answers) == 2 and
            len(manager._streaming_frozen_generated) == 2 and
            len(mock_rm.submissions) == 1 and
            'generated_answer' in mock_rm.submissions[0]['samples_data'][0]
        )

        print(f"\n  총 소요 시간: {total_time:.2f}초")

        if success:
            print("✅ 백그라운드 파이프라인 시뮬레이션 테스트 통과")
            print()
            return True
        else:
            print("❌ 테스트 실패")
            return False

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_prompts_concurrent():
    """다중 프롬프트 동시 처리 테스트"""
    print("=" * 60)
    print("3. 다중 프롬프트 동시 처리 테스트")
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
            search_url="http://localhost:5002/search",
            frozen_model="qwen2.5-vl-72b-instruct",
            frozen_max_tokens=64,
            frozen_max_concurrent=30,
        )

        manager = LLMGenerationManager(
            processor=MockProcessor(),
            actor_rollout_wg=None,
            config=config,
            is_validation=False,
            streaming_reward_manager=None
        )

        # 3개 프롬프트 시뮬레이션 (각 2개 샘플)
        num_prompts = 3
        samples_per_prompt = 2

        manager.questions = [
            "What is 1+1?", "What is 2+2?",  # Prompt 0
            "What is 3+3?", "What is 4+4?",  # Prompt 1
            "What is 5+5?", "What is 6+6?",  # Prompt 2
        ]
        manager.retrievaled_images = [[] for _ in range(6)]
        manager.cropped_images = [[] for _ in range(6)]
        manager.generated_answers = {}
        manager._streaming_frozen_generated = set()
        manager._pending_threads = []

        # Mock streaming reward manager
        class MockStreamingRewardManager:
            def __init__(self):
                self.submissions = []
                self.lock = threading.Lock()

            def submit_prompt(self, uid, sample_indices, samples_data):
                with self.lock:
                    self.submissions.append({
                        'uid': uid,
                        'sample_indices': sample_indices,
                        'samples_data': samples_data
                    })
                print(f"  [Mock] 제출: {uid}")

        mock_rm = MockStreamingRewardManager()
        manager.streaming_reward_manager = mock_rm

        # 동시에 3개 프롬프트 처리
        print("  3개 프롬프트 동시 처리 시작...")
        start_time = time.perf_counter()

        for p_idx in range(num_prompts):
            base_idx = p_idx * samples_per_prompt
            indices = list(range(base_idx, base_idx + samples_per_prompt))
            prompt_id = f"prompt_{p_idx}"
            status = {'completed_samples': 2, 'total_samples': 2, 'submitted': False}

            thread = threading.Thread(
                target=manager._process_prompt_background,
                args=(indices, prompt_id, status),
                daemon=True,
                name=f"FrozenGen-{prompt_id}"
            )
            thread.start()
            manager._pending_threads.append(thread)

        spawn_time = time.perf_counter() - start_time
        print(f"  스레드 시작 시간: {spawn_time:.4f}초")

        # 모든 스레드 완료 대기
        print("  백그라운드 스레드 완료 대기 중...")
        for t in manager._pending_threads:
            t.join(timeout=60)

        total_time = time.perf_counter() - start_time

        # 결과 검증
        print("\n  결과:")
        print(f"    총 생성된 답변: {len(manager.generated_answers)}")
        print(f"    처리 완료된 샘플: {len(manager._streaming_frozen_generated)}")
        print(f"    제출된 프롬프트: {len(mock_rm.submissions)}")

        for idx in sorted(manager.generated_answers.keys()):
            answer = manager.generated_answers[idx][:30] + "..." if len(manager.generated_answers[idx]) > 30 else manager.generated_answers[idx]
            print(f"    [{idx}] {manager.questions[idx]} → {answer}")

        print(f"\n  총 소요 시간: {total_time:.2f}초")
        print(f"  병렬 처리 효과: {samples_per_prompt * num_prompts}개 샘플을 {total_time:.2f}초에 처리")

        # 검증
        success = (
            len(manager.generated_answers) == 6 and
            len(mock_rm.submissions) == 3
        )

        if success:
            print("✅ 다중 프롬프트 동시 처리 테스트 통과")
            print()
            return True
        else:
            print("❌ 테스트 실패")
            return False

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("  Phase 6 통합 테스트 시작")
    print("=" * 60 + "\n")

    results = {}

    # 1. Frozen Generator 배치 호출
    results['frozen_batch'] = test_frozen_generator_batch_call()

    # 2. 백그라운드 파이프라인 시뮬레이션
    results['background_pipeline'] = test_background_pipeline_simulation()

    # 3. 다중 프롬프트 동시 처리
    results['concurrent_prompts'] = test_multiple_prompts_concurrent()

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
        print("🎉 모든 Phase 6 통합 테스트 통과!")
        print("\n완전 비동기 스트리밍 아키텍처가 올바르게 구현되었습니다:")
        print("  - 백그라운드 스레드에서 Frozen Generator 호출 ✓")
        print("  - Thread-safe한 결과 저장 ✓")
        print("  - generated_answer가 samples_data에 포함 ✓")
        print("  - 다중 프롬프트 병렬 처리 ✓")
    else:
        print("⚠️ 일부 테스트 실패")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
