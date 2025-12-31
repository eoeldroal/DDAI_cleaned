import os
import asyncio
import time
import base64
import json
import random
from openai import AsyncOpenAI
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# 병렬 테스트 설정
MAX_CONCURRENT = 128  # 동시 요청 수
TOTAL_REQUESTS = 128  # 총 요청 수

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def run_single_test(client, model, effort, question, image_path):
    print(f"\n--- Testing with Effort: {effort} ---")
    
    # 이미지 인코딩
    base64_image = encode_image(image_path)
    
    # 입력 구성 (generation.py 스타일)
    sys_prompt = (
        "You are a visual QA generator. "
        "Use only the provided images and the user question. "
        "Return ONLY the final answer text without extra explanations."
    )
    
    user_content = [
        {
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{base64_image}"
        },
        {
            "type": "input_text",
            "text": f"Question: {question}"
        }
    ]
    
    inputs = [
        {"role": "developer", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]

    start_time = time.perf_counter()
    try:
        # effort가 None이면 파라미터를 아예 뺌 (기본값 테스트)
        kwargs = {"model": model, "input": inputs, "max_output_tokens": 1024}
        if effort:
            kwargs["reasoning"] = {"effort": effort}

        response = await client.responses.create(**kwargs)
        
        duration = time.perf_counter() - start_time
        
        # 응답 추출
        answer = getattr(response, "output_text", None)
        if not answer and getattr(response, "output", None):
            answer = str(response.output) # 간략화
            
        usage = getattr(response, "usage", None)
        
        print(f"Status: Success")
        print(f"Time: {duration:.4f}s")
        if usage:
            print(f"Tokens: Total={usage.total_tokens}, Input={usage.input_tokens}, Output={usage.output_tokens}")
            # reasoning_tokens가 있는지 확인 (OpenAI 표준)
            if hasattr(usage, 'output_tokens_details'):
                details = usage.output_tokens_details
                if hasattr(details, 'reasoning_tokens'):
                    print(f"  -> Reasoning Tokens: {details.reasoning_tokens}")
        print(f"Answer Preview: {answer[:100]}...")
        
    except Exception as e:
        print(f"Status: Failed ({e})")
        return None, None, str(e)


async def run_single_request(
    client: AsyncOpenAI,
    model: str,
    effort: str,
    question: str,
    image_path: str,
    semaphore: asyncio.Semaphore,
    request_id: int,
) -> dict:
    """
    실제 generation.py 패턴을 따르는 단일 요청 (세마포어 포함)
    """
    async with semaphore:
        base64_image = encode_image(image_path)

        sys_prompt = (
            "You are a visual QA generator. "
            "Use only the provided images and the user question. "
            "Return ONLY the final answer text without extra explanations."
        )

        user_content = [
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{base64_image}"
            },
            {
                "type": "input_text",
                "text": f"Question: {question}"
            }
        ]

        inputs = [
            {"role": "developer", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]

        start_time = time.perf_counter()
        try:
            kwargs = {"model": model, "input": inputs, "max_output_tokens": 1024}
            if effort:
                kwargs["reasoning"] = {"effort": effort}

            response = await client.responses.create(**kwargs)
            duration = time.perf_counter() - start_time

            answer = getattr(response, "output_text", None)
            usage = getattr(response, "usage", None)

            reasoning_tokens = 0
            if usage and hasattr(usage, 'output_tokens_details'):
                details = usage.output_tokens_details
                if hasattr(details, 'reasoning_tokens'):
                    reasoning_tokens = details.reasoning_tokens

            return {
                "request_id": request_id,
                "status": "success",
                "duration": duration,
                "total_tokens": usage.total_tokens if usage else 0,
                "input_tokens": usage.input_tokens if usage else 0,
                "output_tokens": usage.output_tokens if usage else 0,
                "reasoning_tokens": reasoning_tokens,
                "answer_preview": (answer[:50] + "...") if answer and len(answer) > 50 else answer,
            }
        except Exception as e:
            duration = time.perf_counter() - start_time
            return {
                "request_id": request_id,
                "status": "failed",
                "duration": duration,
                "error": str(e),
            }


async def run_parallel_test(client: AsyncOpenAI, model: str, effort: str, test_samples: list):
    """
    generation.py의 asyncio.gather() 패턴을 따르는 병렬 테스트
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    print(f"\n{'='*60}")
    print(f"🚀 병렬 요청 테스트 시작")
    print(f"   총 요청 수: {len(test_samples)}")
    print(f"   동시 요청 수 (세마포어): {MAX_CONCURRENT}")
    print(f"   Reasoning Effort: {effort}")
    print(f"{'='*60}")

    # 태스크 생성 (generation.py 패턴)
    tasks = [
        run_single_request(
            client=client,
            model=model,
            effort=effort,
            question=sample["question"],
            image_path=sample["image_path"],
            semaphore=semaphore,
            request_id=i,
        )
        for i, sample in enumerate(test_samples)
    ]

    # 전체 시간 측정
    total_start = time.perf_counter()

    # asyncio.gather로 병렬 처리 (generation.py 패턴)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    total_duration = time.perf_counter() - total_start

    # 결과 분석
    success_count = 0
    failed_count = 0
    durations = []
    total_tokens_sum = 0
    reasoning_tokens_sum = 0

    for r in results:
        if isinstance(r, Exception):
            failed_count += 1
        elif isinstance(r, dict):
            if r.get("status") == "success":
                success_count += 1
                durations.append(r["duration"])
                total_tokens_sum += r.get("total_tokens", 0)
                reasoning_tokens_sum += r.get("reasoning_tokens", 0)
            else:
                failed_count += 1

    # 통계 출력
    print(f"\n📊 결과 요약:")
    print(f"   성공: {success_count}/{len(test_samples)}")
    print(f"   실패: {failed_count}/{len(test_samples)}")
    print(f"\n⏱️  시간 통계:")
    print(f"   전체 소요 시간: {total_duration:.2f}초")
    if durations:
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        print(f"   개별 요청 평균: {avg_duration:.2f}초")
        print(f"   개별 요청 최소: {min_duration:.2f}초")
        print(f"   개별 요청 최대: {max_duration:.2f}초")
        print(f"   처리량: {len(test_samples) / total_duration:.2f} 요청/초")

    print(f"\n🔢 토큰 통계:")
    print(f"   총 토큰: {total_tokens_sum:,}")
    print(f"   Reasoning 토큰: {reasoning_tokens_sum:,}")
    if success_count > 0:
        print(f"   평균 토큰/요청: {total_tokens_sum / success_count:.0f}")
        print(f"   평균 Reasoning 토큰/요청: {reasoning_tokens_sum / success_count:.0f}")

    return {
        "total_duration": total_duration,
        "success_count": success_count,
        "failed_count": failed_count,
        "avg_duration": sum(durations) / len(durations) if durations else 0,
        "throughput": len(test_samples) / total_duration if total_duration > 0 else 0,
        "total_tokens": total_tokens_sum,
        "reasoning_tokens": reasoning_tokens_sum,
    }


# 분석에서 발견한 문제 케이스들
PROBLEM_CASES = [
    {
        "name": "Case 1: Hallucination - 이미지에 없는 정보",
        "question": "What are the types of processing used by Uber?",
        "image_path": "./search_engine/corpus/img/8273_8.jpg",  # UBER 관련 이미지
        "expected_issue": "hallucination",
        "note": "이미지에 처리 유형 정보가 없으면 'I don't know'라고 해야 함"
    },
    {
        "name": "Case 2: 수치 추출 - 표에서 값 추출",
        "question": 'According to the table on talent adaptability score, what is the difference in the "Average number of employees" between France and Australia?',
        "image_path": "./search_engine/corpus/img/4426_7.jpg",
        "expected_issue": "numeric_extraction",
        "note": "정확한 수치 계산 필요 (0.2가 정답)"
    },
    {
        "name": "Case 3: 복잡한 조건부 질문",
        "question": "What percentage of those surveyed did not report being a Housewife or a Student?",
        "image_path": "./search_engine/corpus/img/1084_7.jpg",  # 베트남 오토바이 소유율 설문
        "expected_issue": "wrong_context",
        "note": "이미지가 질문과 관련 없을 때 어떻게 답하는가"
    },
    {
        "name": "Case 4: 차트 해석",
        "question": "Which Altcoin has the highest value market cap?",
        "image_path": "./search_engine/corpus/img/8484_7.jpg",  # Cryptocurrency 관련
        "expected_issue": "concept_understanding",
        "note": "Bitcoin은 Altcoin이 아님 - 개념 이해 필요"
    },
    {
        "name": "Case 5: 색상 인식",
        "question": "What is the background color of the two credit cards that are visible?",
        "image_path": "./search_engine/corpus/img/6263_2.jpg",  # 신용카드 이미지
        "expected_issue": "visual_recognition",
        "note": "시각적 요소 정확히 인식하는가"
    },
]


def load_samples_from_log(jsonl_path: str, num_samples: int = 128) -> list:
    """
    frozen_generator_detail.jsonl에서 실제 샘플 로드
    """
    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if len(samples) >= num_samples:
                break
            try:
                record = json.loads(line.strip())
                question = record.get("question", "").replace("\nassistant\n", "").strip()
                image_paths = record.get("image_paths", [])

                # 실제 존재하는 이미지 경로만 사용
                valid_paths = [p for p in image_paths if os.path.exists(p)]
                if valid_paths and question:
                    samples.append({
                        "question": question,
                        "image_path": valid_paths[0],  # 첫 번째 이미지 사용
                    })
            except json.JSONDecodeError:
                continue
    return samples


async def main():
    import sys

    # 설정 로드
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = "gpt-5-mini-2025-08-07"

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # 커맨드라인 인자 확인
    if len(sys.argv) > 1 and sys.argv[1] == "--parallel":
        # 128개 병렬 테스트
        print("=" * 60)
        print("GPT-5mini 128개 병렬 요청 테스트")
        print("Reasoning Effort: medium")
        print("=" * 60)

        # frozen_generator_detail.jsonl에서 샘플 로드
        jsonl_path = "./logs/frozen_generator_detail.jsonl"
        if not os.path.exists(jsonl_path):
            print(f"Error: {jsonl_path} not found")
            return

        samples = load_samples_from_log(jsonl_path, TOTAL_REQUESTS)
        print(f"📂 로드된 샘플 수: {len(samples)}")

        if len(samples) < TOTAL_REQUESTS:
            print(f"⚠️ 요청된 {TOTAL_REQUESTS}개보다 적은 {len(samples)}개만 로드됨")

        # 병렬 테스트 실행
        result = await run_parallel_test(client, model, "medium", samples)

        # GRPO 훈련 시간 예측
        print(f"\n{'='*60}")
        print("📈 GRPO 훈련 시간 예측 (Reasoning Effort = medium 기준)")
        print(f"{'='*60}")

        # 가정: GRPO는 각 샘플당 4개 응답 생성 (n_agent=4)
        n_agent = 4
        total_samples = 6277  # frozen_generator_detail.jsonl 전체

        # 예측
        throughput = result["throughput"]
        avg_duration = result["avg_duration"]

        single_pass_requests = total_samples * n_agent
        estimated_time_sec = single_pass_requests / throughput if throughput > 0 else 0
        estimated_time_min = estimated_time_sec / 60

        print(f"   테스트 처리량: {throughput:.2f} 요청/초")
        print(f"   GRPO 단일 패스 요청 수: {single_pass_requests:,} ({total_samples} x {n_agent})")
        print(f"   예상 단일 패스 소요 시간: {estimated_time_min:.1f}분")
        print(f"   (순차 처리 대비 {avg_duration * single_pass_requests / 60:.0f}분 -> {estimated_time_min:.1f}분)")

    else:
        # 개별 문제 케이스 테스트
        print("=" * 60)
        print("GPT-5mini Reasoning Effort 테스트 (Medium)")
        print("분석에서 발견된 문제 케이스 검증")
        print("=" * 60)
        print("\n💡 128개 병렬 테스트를 실행하려면: python tests/test_frozen_effort.py --parallel")

        for case in PROBLEM_CASES:
            print(f"\n{'='*60}")
            print(f"📋 {case['name']}")
            print(f"   질문: {case['question'][:80]}...")
            print(f"   이미지: {case['image_path']}")
            print(f"   예상 이슈: {case['expected_issue']}")
            print(f"   노트: {case['note']}")
            print("-" * 60)

            if not os.path.exists(case['image_path']):
                print(f"   ⚠️ 이미지 파일 없음, 건너뜀")
                continue

            await run_single_test(
                client,
                model,
                "medium",  # reasoning effort = medium
                case['question'],
                case['image_path']
            )

            # API rate limit 방지
            await asyncio.sleep(1)

        print("\n" + "=" * 60)
        print("테스트 완료")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
