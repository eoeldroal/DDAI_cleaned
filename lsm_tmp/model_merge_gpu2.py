import os
import torch
from collections import defaultdict
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

# DTensor 포장 벗기기 (이전과 동일)
def unwrap_dtensor(tensor):
    if type(tensor).__name__ == 'DTensor' or hasattr(tensor, 'to_local'):
        return tensor.to_local()
    return tensor

def merge_grpo_checkpoint(base_path, output_path):
    config_path = os.path.join(base_path, "actor/huggingface")
    
    # 1. 설정 로드
    print(f"🛠️ [1단계] Config 로드: {config_path}")
    try:
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    except Exception as e:
        print(f"   ❌ Config 로드 실패: {e}")
        return

    # 2. 빈 모델 생성
    print(f"🏗️ [2단계] 빈 모델 생성 중...")
    with torch.device("cpu"):
        model = AutoModelForVision2Seq.from_config(config, trust_remote_code=True)
    model.to(dtype=torch.bfloat16)

    # 3. 가중치 수집 (이어붙이기 준비)
    # [수정됨] 4개 -> 2개로 로그 메시지 변경
    print("🧩 [3단계] 2개 GPU의 파라미터 조각을 수집합니다 (메모리 주의)...")
    
    shards = defaultdict(list)
    
    # [수정됨] range(4) -> range(2)로 변경 (0, 1번만 반복)
    for rank in range(2):
        # [수정됨] 파일명 포맷 변경: world_size_4 -> world_size_2
        checkpoint_name = f"model_world_size_2_rank_{rank}.pt"
        checkpoint_path = os.path.join(base_path, "actor", checkpoint_name)
        
        print(f"   ㄴ 📂 Rank {rank} 로드 중...")
        if not os.path.exists(checkpoint_path):
            print(f"      ❌ 파일 없음: {checkpoint_path}")
            return

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        for key, tensor in state_dict.items():
            # DTensor 포장 벗기기
            clean_tensor = unwrap_dtensor(tensor)
            shards[key].append(clean_tensor)

    # 4. 조각 이어붙이기 (Concatenate)
    print("✨ [4단계] 수집된 조각들을 하나로 합칩니다 (Concatenate)...")
    full_state_dict = {}
    
    for key, tensor_list in shards.items():
        try:
            # 로그를 볼 때 모든 파라미터가 0번 차원(dim=0)으로 쪼개져 있음
            merged_tensor = torch.cat(tensor_list, dim=0)
            full_state_dict[key] = merged_tensor
        except Exception as e:
            print(f"   ⚠️ 병합 실패 ({key}): {e} -> 첫 번째 조각만 사용 시도")
            full_state_dict[key] = tensor_list[0]

    # 5. 모델에 주입
    print("💉 [5단계] 완성된 가중치를 모델에 주입합니다...")
    
    missing, unexpected = model.load_state_dict(full_state_dict, strict=False)
    
    if missing:
        print(f"   ⚠️ 누락된 키: {len(missing)}개")
    if unexpected:
        print(f"   ℹ️ 불필요한 키: {len(unexpected)}개")

    # 6. 저장
    print(f"💾 [6단계] '{output_path}'에 저장 중...")
    model.save_pretrained(output_path)
    
    try:
        processor = AutoProcessor.from_pretrained(config_path, trust_remote_code=True)
        processor.save_pretrained(output_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(config_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)

    print(f"✅ 병합 완료! 확인해보세요: {output_path}")

# --- 실행부 ---
input_folder = "/data/daedong/gspo_phase1/gspo_phase1_last" 
output_folder = "./RL_results/merged_gspo_phase1"

if __name__ == "__main__":
    merge_grpo_checkpoint(input_folder, output_folder)