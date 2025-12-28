import os
import multiprocessing
import math
from tqdm import tqdm
import time
import sys

# CUDA re-initialization 방지를 위해 spawn 방식 설정
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# 라이브러리 경로 설정
sys.path.append(os.path.join(os.getcwd(), 'search_engine'))

def run_ingestion_on_gpu(gpu_id, files_subset, dataset_dir, output_prefix, batch_size=16):
    """특정 GPU에서 이미지 배치를 처리하는 함수"""
    # 자식 프로세스 내부에서만 Ingestion을 임포트하여 초기화 충돌 방지
    from ingestion import Ingestion
    
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Initializing with {len(files_subset)} files (Batch size: {batch_size})")
    
    try:
        # Ingestion 인스턴스 생성 (해당 GPU 할당)
        ingestion = Ingestion(
            dataset_dir=dataset_dir,
            input_prefix='img',
            output_prefix=output_prefix,
            device=device
        )
        
        # 전체 파일을 배치 단위로 쪼개기
        for i in tqdm(range(0, len(files_subset), batch_size), desc=f"GPU {gpu_id}", position=gpu_id):
            batch = files_subset[i:i + batch_size]
            # 이미 처리된 파일 제외
            batch_to_do = [(inp, out) for inp, out in batch if not os.path.exists(out)]
            
            if batch_to_do:
                ingestion.ingestion_batch(batch_to_do)
                
    except Exception as e:
        print(f"[GPU {gpu_id}] Fatal Error: {e}")

if __name__ == '__main__':
    dataset_dir = './search_engine/corpus'
    input_dir = os.path.join(dataset_dir, 'img')
    output_prefix = 'colqwen_ingestion'
    output_dir = os.path.join(dataset_dir, output_prefix)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 처리해야 할 파일 목록 수집
    print("Scanning files...")
    all_files = sorted(os.listdir(input_dir))
    file_to_process = []
    for file in all_files:
        file_prefix, ext = os.path.splitext(file)
        if ext.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
            continue
        input_file = os.path.join(input_dir, file)
        output_file = os.path.join(output_dir, file_prefix) + '.node'
        if not os.path.exists(output_file):
            file_to_process.append((input_file, output_file))
            
    total_count = len(file_to_process)
    print(f"Total files to process: {total_count}")
    
    if total_count == 0:
        print("No files to process.")
        exit()
        
    # 2. GPU 개수에 맞춰 파일 목록 분할
    num_gpus = 8
    chunk_size = math.ceil(total_count / num_gpus)
    chunks = [file_to_process[i:i + chunk_size] for i in range(0, total_count, chunk_size)]
    
    # 3. 각 GPU별로 프로세스 생성 및 실행
    processes = []
    print(f"Launching {len(chunks)} GPU processes using 'spawn' method...")
    for i in range(len(chunks)):
        p = multiprocessing.Process(
            target=run_ingestion_on_gpu,
            args=(i, chunks[i], dataset_dir, output_prefix, 16)
        )
        p.start()
        processes.append(p)
        time.sleep(5)
        
    # 4. 모든 프로세스 종료 대기
    for p in processes:
        p.join()
        
    print("Aggressive ingestion completed.")
