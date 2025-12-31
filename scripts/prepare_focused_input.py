import pandas as pd
import json
import os
import glob

def load_hard_samples_from_logs(log_path, original_data_path, threshold=0.4):
    """
    1. 로그에서 UID별 평균 점수 계산
    2. threshold 이하인 UID 추출
    3. 원본 데이터에서 해당 UID의 데이터 로드
    """
    print(f"Loading logs from {log_path}...")
    data = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    
    df_log = pd.DataFrame(data)
    
    # UID별 평균 점수 계산
    # 'uid'가 없으면 에러가 나겠지만, 스키마 확인했으므로 있다고 가정
    if 'uid' not in df_log.columns or 'final_score' not in df_log.columns:
        print("Error: 'uid' or 'final_score' not found in logs.")
        return pd.DataFrame()
    
    # Group by UID and calculate mean score
    df_grouped = df_log.groupby('uid')['final_score'].mean().reset_index()
    
    # Filter by threshold
    hard_uids_df = df_grouped[df_grouped['final_score'] <= threshold]
    hard_uids = set(hard_uids_df['uid'].tolist())
    
    print(f"Total Unique UIDs in logs: {len(df_grouped)}")
    print(f"Hard UIDs (Avg Score <= {threshold}): {len(hard_uids)}")
    
    # Load original data to get full content (images, etc.)
    # 원본 데이터 위치 추정 (curriculum_bucket_a.parquet가 Phase 1 학습 데이터였음)
    # 만약 파일이 없다면 다른 경로를 찾아야 함
    if not os.path.exists(original_data_path):
        print(f"Warning: Original data path {original_data_path} not found. Trying to find alternative...")
        # 대안 경로 로직은 필요시 추가
        return pd.DataFrame()
        
    print(f"Loading original data from {original_data_path}...")
    df_orig = pd.read_parquet(original_data_path)
    
    # Filter original data with hard_uids
    # uid 컬럼 매칭 확인: 로그는 'uid', 원본은 'id'
    if 'id' in df_orig.columns:
        df_hard_a = df_orig[df_orig['id'].isin(hard_uids)].copy()
    elif 'uid' in df_orig.columns:
        df_hard_a = df_orig[df_orig['uid'].isin(hard_uids)].copy()
    else:
        print(f"Error: Neither 'id' nor 'uid' found in original columns: {df_orig.columns}")
        return pd.DataFrame()

    print(f"Recovered Hard A samples: {len(df_hard_a)}")
    
    return df_hard_a

def load_bucket_0(bucket_0_path):
    print(f"Loading Bucket 0 from {bucket_0_path}...")
    if not os.path.exists(bucket_0_path):
        print(f"Error: Bucket 0 file not found at {bucket_0_path}")
        return pd.DataFrame()
        
    df_b0 = pd.read_parquet(bucket_0_path)
    print(f"Bucket 0 samples: {len(df_b0)}")
    return df_b0

def main():
    # Configuration
    LOG_PATH = "logs/flash_rm_detail.jsonl"
    ORIGINAL_DATA_PATH = "data/curriculum_bucket_a.parquet" # Phase 1/2의 Main Dataset
    BUCKET_0_PATH = "/opt/dlami/nvme/isdslab/HyunBin/DDAI_cleaned/data/curriculum_bucket_0_filtered.parquet"
    OUTPUT_PATH = "data/focused_round1.parquet"
    
    # 1. Load Hard A
    df_hard_a = load_hard_samples_from_logs(LOG_PATH, ORIGINAL_DATA_PATH, threshold=0.2)
    
    # 2. Load Bucket 0
    df_bucket_0 = load_bucket_0(BUCKET_0_PATH)
    
    if df_hard_a.empty and df_bucket_0.empty:
        print("No data found for Focused RL. Exiting.")
        return

    # 3. Align Columns (if necessary)
    # 두 데이터프레임의 컬럼이 일치해야 병합 가능
    print("Aligning columns...")
    common_cols = list(set(df_hard_a.columns) & set(df_bucket_0.columns))
    print(f"Common columns: {common_cols}")
    
    # 필수 컬럼(images, question, answer/ground_truth)이 있는지 확인
    # 없다면 최대한 살리는 방향으로
    
    # 4. Merge
    df_merged = pd.concat([df_hard_a, df_bucket_0], ignore_index=True)
    
    # 5. Deduplicate
    initial_len = len(df_merged)
    if 'uid' in df_merged.columns:
        df_merged = df_merged.drop_duplicates(subset=['uid'])
    else:
        # uid가 없으면 질문 텍스트로 중복 제거 시도
        # 컬럼명 확인 필요 (question, query, text 등)
        text_col = 'question' if 'question' in df_merged.columns else 'query'
        if text_col in df_merged.columns:
            df_merged = df_merged.drop_duplicates(subset=[text_col])
            
    print(f"Merged & Deduplicated: {initial_len} -> {len(df_merged)}")
    
    # 6. Save
    print(f"Saving to {OUTPUT_PATH}...")
    df_merged.to_parquet(OUTPUT_PATH)
    print("Done.")

if __name__ == "__main__":
    main()
