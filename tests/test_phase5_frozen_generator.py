#!/usr/bin/env python3
"""
Phase 5 Frozen Generator 테스트

이 테스트는 OpenAI AsyncClient를 사용한 Frozen Generator의
비동기 처리가 올바르게 동작하는지 확인합니다.
"""

import os
import sys
import asyncio
import time

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# .env 로드
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))


def test_env_variables():
    """환경 변수 확인"""
    print("=" * 60)
    print("1. 환경 변수 테스트")
    print("=" * 60)

    dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    assert dashscope_key, "DASHSCOPE_API_KEY가 설정되지 않았습니다!"
    assert gemini_key, "GEMINI_API_KEY가 설정되지 않았습니다!"

    print(f"✅ DASHSCOPE_API_KEY: {dashscope_key[:10]}...")
    print(f"✅ GEMINI_API_KEY: {gemini_key[:10]}...")
    print()


def test_openai_async_client_import():
    """OpenAI AsyncClient 임포트 테스트"""
    print("=" * 60)
    print("2. OpenAI AsyncClient 임포트 테스트")
    print("=" * 60)

    try:
        from openai import AsyncOpenAI
        print("✅ OpenAI AsyncClient 임포트 성공")

        # 클라이언트 생성 테스트
        client = AsyncOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            timeout=60.0,
            max_retries=0,
        )
        print(f"✅ AsyncOpenAI 클라이언트 생성 성공: {type(client)}")
        print()
        return client
    except ImportError as e:
        print(f"❌ OpenAI SDK 임포트 실패: {e}")
        return None


async def test_simple_api_call(client):
    """간단한 API 호출 테스트 (텍스트만)"""
    print("=" * 60)
    print("3. 간단한 API 호출 테스트 (텍스트만)")
    print("=" * 60)

    try:
        start_time = time.perf_counter()

        response = await client.chat.completions.create(
            model="qwen2.5-vl-72b-instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2 + 2? Answer with just the number."}
            ],
            max_tokens=50,
            temperature=0.1,
        )

        elapsed = time.perf_counter() - start_time

        if response.choices and len(response.choices) > 0:
            answer = response.choices[0].message.content
            print(f"✅ API 호출 성공!")
            print(f"   응답: {answer}")
            print(f"   소요 시간: {elapsed:.2f}초")
            print()
            return True
        else:
            print("❌ 응답이 없습니다")
            return False

    except Exception as e:
        print(f"❌ API 호출 실패: {e}")
        return False


async def test_concurrent_calls(client, num_calls=5):
    """동시 API 호출 테스트"""
    print("=" * 60)
    print(f"4. 동시 API 호출 테스트 ({num_calls}개)")
    print("=" * 60)

    questions = [
        "What is 1 + 1?",
        "What is 2 + 2?",
        "What is 3 + 3?",
        "What is 4 + 4?",
        "What is 5 + 5?",
    ][:num_calls]

    async def single_call(q):
        try:
            response = await client.chat.completions.create(
                model="qwen2.5-vl-72b-instruct",
                messages=[
                    {"role": "system", "content": "Answer with just the number."},
                    {"role": "user", "content": q}
                ],
                max_tokens=20,
                temperature=0.1,
            )
            if response.choices:
                return response.choices[0].message.content.strip()
            return ""
        except Exception as e:
            return f"Error: {e}"

    start_time = time.perf_counter()

    # 동시 실행
    tasks = [single_call(q) for q in questions]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.perf_counter() - start_time

    print("결과:")
    for q, r in zip(questions, results):
        status = "✅" if not isinstance(r, Exception) and "Error" not in str(r) else "❌"
        print(f"  {status} {q} → {r}")

    print(f"\n총 소요 시간: {elapsed:.2f}초")
    print(f"평균 시간: {elapsed/num_calls:.2f}초/요청")
    print(f"처리량: {num_calls/elapsed:.2f} req/s")
    print()

    return all(not isinstance(r, Exception) and "Error" not in str(r) for r in results)


def test_generation_module_import():
    """generation.py 모듈 임포트 테스트"""
    print("=" * 60)
    print("5. generation.py 모듈 Phase 5 구성요소 테스트")
    print("=" * 60)

    try:
        # 직접 임포트 대신 필요한 부분만 테스트
        from vrag_agent.generation import (
            _HAS_OPENAI_ASYNC,
            _OPENAI_ASYNC_CLIENT,
            _image_to_base64_url,
            _call_frozen_generator_async_single,
        )

        print(f"✅ _HAS_OPENAI_ASYNC: {_HAS_OPENAI_ASYNC}")
        print(f"✅ _OPENAI_ASYNC_CLIENT: {type(_OPENAI_ASYNC_CLIENT)}")
        print(f"✅ _image_to_base64_url 함수: {_image_to_base64_url}")
        print(f"✅ _call_frozen_generator_async_single 함수: {_call_frozen_generator_async_single}")
        print()
        return True

    except ImportError as e:
        print(f"❌ 임포트 실패: {e}")
        return False


async def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("  Phase 5 Frozen Generator 테스트 시작")
    print("=" * 60 + "\n")

    results = {}

    # 1. 환경 변수 테스트
    try:
        test_env_variables()
        results['env'] = True
    except AssertionError as e:
        print(f"❌ {e}")
        results['env'] = False

    # 2. OpenAI AsyncClient 임포트 테스트
    client = test_openai_async_client_import()
    results['import'] = client is not None

    if client:
        # 3. 간단한 API 호출 테스트
        results['simple_call'] = await test_simple_api_call(client)

        # 4. 동시 API 호출 테스트
        results['concurrent'] = await test_concurrent_calls(client, num_calls=5)
    else:
        results['simple_call'] = False
        results['concurrent'] = False

    # 5. generation.py 모듈 테스트
    results['module'] = test_generation_module_import()

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
        print("🎉 모든 테스트 통과!")
    else:
        print("⚠️ 일부 테스트 실패")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
