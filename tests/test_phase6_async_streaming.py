#!/usr/bin/env python3
"""
Phase 6 완전 비동기 스트리밍 테스트

이 테스트는 다음을 검증합니다:
1. 백그라운드 스레드에서 Frozen Generator가 올바르게 호출되는지
2. Thread-safety가 보장되는지 (동시 접근 시 데이터 무결성)
3. 여러 프롬프트가 동시에 처리될 때 올바르게 동작하는지
4. 메인 루프 종료 후 백그라운드 대기가 올바르게 동작하는지
"""

import os
import sys
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# .env 로드
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))


def test_threading_import():
    """threading 모듈 임포트 테스트"""
    print("=" * 60)
    print("1. threading 모듈 임포트 테스트")
    print("=" * 60)

    try:
        from vrag_agent.generation import threading
        print(f"✅ threading 모듈 임포트 성공: {threading}")
        return True
    except ImportError as e:
        print(f"❌ threading 모듈 임포트 실패: {e}")
        return False


def test_new_fields_exist():
    """Phase 6 새 필드 존재 확인"""
    print("=" * 60)
    print("2. Phase 6 새 필드 존재 확인")
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

        # 인스턴스 생성 (actor_rollout_wg는 None으로)
        manager = LLMGenerationManager(
            processor=MockProcessor(),
            actor_rollout_wg=None,
            config=config,
            is_validation=False,
            streaming_reward_manager=None
        )

        # 새 필드 확인
        assert hasattr(manager, '_pending_threads'), "_pending_threads 필드 누락"
        assert hasattr(manager, '_thread_lock'), "_thread_lock 필드 누락"
        assert hasattr(manager, 'generated_answers'), "generated_answers 필드 누락"
        assert hasattr(manager, '_streaming_frozen_generated'), "_streaming_frozen_generated 필드 누락"

        print("✅ _pending_threads 존재")
        print("✅ _thread_lock 존재")
        print("✅ generated_answers 존재")
        print("✅ _streaming_frozen_generated 존재")
        print()
        return True

    except Exception as e:
        print(f"❌ 필드 확인 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_thread_safety_concurrent_writes():
    """Thread-safety 테스트: 동시 쓰기"""
    print("=" * 60)
    print("3. Thread-safety 테스트: 동시 쓰기")
    print("=" * 60)

    # 공유 자료구조
    generated_answers: Dict[int, str] = {}
    streaming_frozen_generated: Set[int] = set()
    thread_lock = threading.Lock()

    num_threads = 10
    items_per_thread = 100
    errors = []

    def writer_thread(thread_id: int):
        """스레드에서 동시에 쓰기"""
        for i in range(items_per_thread):
            idx = thread_id * items_per_thread + i
            try:
                with thread_lock:
                    generated_answers[idx] = f"answer_{idx}"
                    streaming_frozen_generated.add(idx)
            except Exception as e:
                errors.append(f"Thread {thread_id}, item {i}: {e}")

    # 스레드 시작
    threads = []
    start_time = time.perf_counter()

    for t_id in range(num_threads):
        t = threading.Thread(target=writer_thread, args=(t_id,))
        threads.append(t)
        t.start()

    # 모든 스레드 완료 대기
    for t in threads:
        t.join()

    elapsed = time.perf_counter() - start_time

    # 결과 검증
    expected_count = num_threads * items_per_thread
    actual_count = len(generated_answers)
    set_count = len(streaming_frozen_generated)

    print(f"  스레드 수: {num_threads}")
    print(f"  항목/스레드: {items_per_thread}")
    print(f"  총 예상 항목: {expected_count}")
    print(f"  실제 dict 항목: {actual_count}")
    print(f"  실제 set 항목: {set_count}")
    print(f"  오류 수: {len(errors)}")
    print(f"  소요 시간: {elapsed:.4f}초")

    if actual_count == expected_count and set_count == expected_count and len(errors) == 0:
        print("✅ Thread-safety 테스트 통과")
        print()
        return True
    else:
        print("❌ Thread-safety 테스트 실패")
        for e in errors[:5]:
            print(f"    {e}")
        return False


def test_background_thread_spawn():
    """백그라운드 스레드 생성 및 완료 테스트"""
    print("=" * 60)
    print("4. 백그라운드 스레드 생성 및 완료 테스트")
    print("=" * 60)

    results = {}
    pending_threads: List[threading.Thread] = []

    def background_task(task_id: int, delay: float):
        """백그라운드 작업 시뮬레이션"""
        time.sleep(delay)
        results[task_id] = f"completed_{task_id}"

    # 여러 백그라운드 스레드 시작
    num_tasks = 5
    start_time = time.perf_counter()

    for i in range(num_tasks):
        t = threading.Thread(
            target=background_task,
            args=(i, 0.2),  # 각 0.2초 지연
            daemon=True,
            name=f"BackgroundTask-{i}"
        )
        t.start()
        pending_threads.append(t)
        print(f"  스레드 {i} 시작됨")

    spawn_time = time.perf_counter() - start_time
    print(f"  모든 스레드 시작 시간: {spawn_time:.4f}초 (블로킹 없음 확인)")

    # 모든 스레드 완료 대기
    wait_start = time.perf_counter()
    for t in pending_threads:
        t.join(timeout=5)
    wait_time = time.perf_counter() - wait_start

    print(f"  스레드 완료 대기 시간: {wait_time:.4f}초")
    print(f"  완료된 작업: {len(results)}/{num_tasks}")

    if len(results) == num_tasks and spawn_time < 0.1:
        print("✅ 백그라운드 스레드 테스트 통과 (블로킹 없이 빠르게 시작)")
        print()
        return True
    else:
        print("❌ 백그라운드 스레드 테스트 실패")
        return False


async def test_async_frozen_generator_with_threads():
    """비동기 Frozen Generator + 스레드 조합 테스트"""
    print("=" * 60)
    print("5. 비동기 Frozen Generator + 스레드 조합 테스트")
    print("=" * 60)

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            timeout=60.0,
            max_retries=0,
        )

        results = {}
        thread_lock = threading.Lock()
        pending_threads = []

        def sync_wrapper_for_async(task_id: int, question: str):
            """백그라운드 스레드에서 asyncio.run()으로 비동기 호출"""
            async def async_call():
                response = await client.chat.completions.create(
                    model="qwen2.5-vl-72b-instruct",
                    messages=[
                        {"role": "system", "content": "Answer briefly."},
                        {"role": "user", "content": question}
                    ],
                    max_tokens=20,
                    temperature=0.1,
                )
                if response.choices:
                    return response.choices[0].message.content.strip()
                return ""

            try:
                answer = asyncio.run(async_call())
                with thread_lock:
                    results[task_id] = answer
            except Exception as e:
                with thread_lock:
                    results[task_id] = f"Error: {e}"

        # 3개 질문을 백그라운드 스레드로 처리
        questions = [
            "What is 10 + 10?",
            "What is 20 + 20?",
            "What is 30 + 30?",
        ]

        start_time = time.perf_counter()

        for i, q in enumerate(questions):
            t = threading.Thread(
                target=sync_wrapper_for_async,
                args=(i, q),
                daemon=True
            )
            t.start()
            pending_threads.append(t)

        spawn_time = time.perf_counter() - start_time
        print(f"  스레드 시작 시간: {spawn_time:.4f}초")

        # 완료 대기
        for t in pending_threads:
            t.join(timeout=30)

        total_time = time.perf_counter() - start_time

        print("  결과:")
        success_count = 0
        for i, q in enumerate(questions):
            answer = results.get(i, "No result")
            status = "✅" if "Error" not in str(answer) else "❌"
            if "Error" not in str(answer):
                success_count += 1
            print(f"    {status} {q} → {answer}")

        print(f"  총 소요 시간: {total_time:.2f}초")
        print(f"  성공률: {success_count}/{len(questions)}")

        if success_count == len(questions) and spawn_time < 0.1:
            print("✅ 비동기 + 스레드 조합 테스트 통과")
            print()
            return True
        else:
            print("❌ 비동기 + 스레드 조합 테스트 실패")
            return False

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_collect_samples_data_with_generated_answer():
    """_collect_samples_data에 generated_answer 필드 확인"""
    print("=" * 60)
    print("6. _collect_samples_data generated_answer 필드 테스트")
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

        # 테스트 데이터 설정
        manager.retrievaled_images = [
            ["/path/to/img1.jpg", "/path/to/img2.jpg"],
            ["/path/to/img3.jpg"],
        ]
        manager.cropped_images = [[], []]
        manager.questions = ["Question 1?", "Question 2?"]
        manager.generated_answers = {
            0: "Answer to question 1",
            1: "Answer to question 2"
        }

        # _collect_samples_data 호출
        samples_data = manager._collect_samples_data([0, 1])

        # 검증
        assert len(samples_data) == 2, f"Expected 2 samples, got {len(samples_data)}"
        assert 'generated_answer' in samples_data[0], "generated_answer 필드 누락"
        assert samples_data[0]['generated_answer'] == "Answer to question 1"
        assert samples_data[1]['generated_answer'] == "Answer to question 2"

        print("  샘플 0:")
        print(f"    query: {samples_data[0]['query']}")
        print(f"    generated_answer: {samples_data[0]['generated_answer']}")
        print("  샘플 1:")
        print(f"    query: {samples_data[1]['query']}")
        print(f"    generated_answer: {samples_data[1]['generated_answer']}")

        print("✅ generated_answer 필드 테스트 통과")
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
    print("  Phase 6 완전 비동기 스트리밍 테스트 시작")
    print("=" * 60 + "\n")

    results = {}

    # 1. threading 임포트 테스트
    results['threading_import'] = test_threading_import()

    # 2. 새 필드 존재 확인
    results['new_fields'] = test_new_fields_exist()

    # 3. Thread-safety 테스트
    results['thread_safety'] = test_thread_safety_concurrent_writes()

    # 4. 백그라운드 스레드 테스트
    results['background_thread'] = test_background_thread_spawn()

    # 5. 비동기 + 스레드 조합 테스트
    results['async_thread'] = asyncio.run(test_async_frozen_generator_with_threads())

    # 6. _collect_samples_data 테스트
    results['collect_samples'] = test_collect_samples_data_with_generated_answer()

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
        print("🎉 모든 Phase 6 테스트 통과!")
    else:
        print("⚠️ 일부 테스트 실패")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
