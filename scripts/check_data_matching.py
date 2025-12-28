import json
import pandas as pd
import re

def extract_uid_from_ref_docs(ref_docs):
    if not ref_docs: return None
    ref_doc = ref_docs[0]
    match = re.match(r"(\d+)_\d+", ref_doc)
    if match:
        doc_id = match.group(1)
        return f"train_{doc_id}"
    return None

def main():
    log_file_path = "logs/gspo_phase1.json"
    train_data_path = "data/rag/slidevqa_train_6667.parquet"

    print(f"Loading training data: {train_data_path}")
    try:
        df_train = pd.read_parquet(train_data_path)
    except FileNotFoundError:
        print("Error: Training data file not found.")
        return

    train_uids = set(df_train['id'])
    print(f"Total samples in training data: {len(train_uids)}")

    print(f"Loading log file: {log_file_path}")
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
    except FileNotFoundError:
        print("Error: Log file not found.")
        return

    log_uids = set()
    uid_scores = {}

    for step_key, step_data in log_data.items():
        for uuid_key, query_data in step_data.items():
            agents = query_data.get("agents", [])
            if not agents: continue
            
            first_agent = agents[0]
            ndcg_details = first_agent.get("scores", {}).get("ndcg_details", {})
            ref_docs = ndcg_details.get("reference_documents", [])
            
            uid = extract_uid_from_ref_docs(ref_docs)
            if uid:
                log_uids.add(uid)
                
                # 점수 계산 (Bucket 분석용)
                ndcg_values = []
                for agent in agents:
                    scores = agent.get("scores", {})
                    ndcg = scores.get("ndcg_value", 0.0)
                    ndcg_values.append(ndcg)
                
                if ndcg_values:
                    avg_ndcg = sum(ndcg_values) / len(ndcg_values)
                    uid_scores[uid] = avg_ndcg

    print(f"Total unique UIDs found in log: {len(log_uids)}")

    # 교집합 및 차집합 분석
    matched_uids = train_uids.intersection(log_uids)
    only_in_train = train_uids - log_uids
    only_in_log = log_uids - train_uids

    print(f"\n[Matching Statistics]")
    print(f"1. Matched (Score available for training): {len(matched_uids)}")
    print(f"2. Only in Training Data (No score in log): {len(only_in_train)}")
    print(f"3. Only in Log (Not in current training set): {len(only_in_log)}")

    # Bucket 분포 (매칭된 데이터 기준)
    buckets = {'A': 0, 'B': 0, '0': 0}
    for uid in matched_uids:
        score = uid_scores[uid]
        if score > 0.7:
            buckets['B'] += 1
        elif 0.1 <= score <= 0.7:
            buckets['A'] += 1
        else:
            buckets['0'] += 1
            
    print(f"\n[Bucket Distribution (Matched Only)]")
    print(f"  - Bucket B (Mastered, > 0.7): {buckets['B']}")
    print(f"  - Bucket A (Target, 0.1 ~ 0.7): {buckets['A']}")
    print(f"  - Bucket 0 (Unsolvable, < 0.1): {buckets['0']}")
    print(f"  - Total: {sum(buckets.values())}")

if __name__ == "__main__":
    main()
