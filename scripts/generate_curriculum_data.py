import json
import pandas as pd
import os
import re
import numpy as np

def extract_uid_from_ref_docs(ref_docs):
    """
    reference_documents 리스트에서 uid를 추출합니다.
    예: ["6391_9"] -> "train_6391"
    """
    if not ref_docs:
        return None
    
    # 첫 번째 문서만 사용
    ref_doc = ref_docs[0]
    
    # 숫자_숫자 형태라고 가정하고 앞부분 숫자만 추출
    # 예: 6391_9 -> 6391
    match = re.match(r"(\d+)_\d+", ref_doc)
    if match:
        doc_id = match.group(1)
        return f"train_{doc_id}"
    
    return None

def update_extra_info(row):
    """
    행(row)의 extra_info 딕셔너리에 phase1_ndcg 점수를 추가합니다.
    """
    extra_info = row['extra_info']
    ndcg_score = row['ndcg_score']
    
    # extra_info가 None이거나 비어있을 수 있음
    if extra_info is None:
        extra_info = {}
    elif isinstance(extra_info, str):
        # 만약 문자열로 저장되어 있다면 파싱 시도 (보통은 dict)
        try:
            extra_info = json.loads(extra_info)
        except:
            extra_info = {}
    elif isinstance(extra_info, np.ndarray):
        extra_info = extra_info.tolist()
        if isinstance(extra_info, list): # array of dicts? usually it's a single dict per row
             # This case depends on how pandas loaded the object column
             pass
    
    # 딕셔너리 복사 (원본 보존)
    new_extra_info = extra_info.copy() if isinstance(extra_info, dict) else {}
    
    # 점수 추가
    new_extra_info['phase1_ndcg'] = float(ndcg_score)
    
    return new_extra_info

def main():
    log_file_path = "logs/gspo_phase1.json"
    train_data_path = "data/rag/slidevqa_train_6667.parquet"
    
    output_paths = {
        'A': "data/curriculum_bucket_a.parquet",
        'B': "data/curriculum_bucket_b.parquet",
        '0': "data/curriculum_bucket_0.parquet"
    }

    print(f"Loading log file: {log_file_path}")
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return

    print(f"Loading training data: {train_data_path}")
    try:
        df_train = pd.read_parquet(train_data_path)
    except FileNotFoundError:
        print(f"Error: Training data file not found at {train_data_path}")
        return

    print(f"Loaded {len(df_train)} training samples.")

    uid_scores = {}

    print("Processing logs...")
    for step_key, step_data in log_data.items():
        for uuid_key, query_data in step_data.items():
            agents = query_data.get("agents", [])
            if not agents: continue

            first_agent = agents[0]
            ndcg_details = first_agent.get("scores", {}).get("ndcg_details", {})
            ref_docs = ndcg_details.get("reference_documents", [])
            
            uid = extract_uid_from_ref_docs(ref_docs)
            if not uid: continue

            ndcg_values = []
            for agent in agents:
                scores = agent.get("scores", {})
                ndcg = scores.get("ndcg_value", 0.0)
                ndcg_values.append(ndcg)
            
            if not ndcg_values: continue
                
            avg_ndcg = sum(ndcg_values) / len(ndcg_values)
            uid_scores[uid] = avg_ndcg

    print(f"Processed scores for {len(uid_scores)} unique UIDs.")

    # Bucketing
    buckets = {'A': [], 'B': [], '0': []}
    
    for uid, score in uid_scores.items():
        if score > 0.7:
            buckets['B'].append((uid, score))
        elif 0.1 <= score <= 0.7:
            buckets['A'].append((uid, score))
        else:
            buckets['0'].append((uid, score))

    # 각 버킷별로 처리 및 저장
    os.makedirs(os.path.dirname(output_paths['A']), exist_ok=True)

    for bucket_name, uids_with_scores in buckets.items():
        if not uids_with_scores:
            continue
            
        print(f"\nProcessing Bucket {bucket_name}...")
        
        score_df = pd.DataFrame(uids_with_scores, columns=['id', 'ndcg_score'])
        merged_df = pd.merge(df_train, score_df, on='id', how='inner')
        merged_df = merged_df.sort_values(by='ndcg_score', ascending=False)
        
        # [수정] extra_info 업데이트
        # apply 함수를 사용하여 각 행의 extra_info에 ndcg_score 추가
        print("  >> Updating extra_info with phase1_ndcg scores...")
        merged_df['extra_info'] = merged_df.apply(update_extra_info, axis=1)
        
        # 점수 컬럼 제거 및 저장
        final_df = merged_df.drop(columns=['ndcg_score'])
        output_path = output_paths[bucket_name]
        
        final_df.to_parquet(output_path, index=False)
        print(f"  >> Saved {len(final_df)} samples to {output_path}")

if __name__ == "__main__":
    main()