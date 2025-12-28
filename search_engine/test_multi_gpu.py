import os
import sys
import multiprocessing
import time

# 라이브러리 경로 설정
sys.path.append(os.path.join(os.getcwd(), 'search_engine'))

# spawn 방식 설정
try:
    multiprocessing.set_start_method('spawn', force=True)
except:
    pass

from ingestion_multi_gpu_aggressive import run_ingestion_on_gpu

if __name__ == '__main__':
    dataset_dir = './search_engine/corpus'
    input_dir = os.path.join(dataset_dir, 'img')
    output_prefix = 'colqwen_ingestion'

    # 이미지 목록 가져오기
    all_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))])
    
    # 8개 GPU에 각각 2개씩 배분
    test_files = []
    for i in range(8):
        gpu_files = []
        for j in range(2):
            idx = i * 2 + j
            if idx < len(all_files):
                file = all_files[idx]
                inp = os.path.join(input_dir, file)
                out = os.path.join(dataset_dir, output_prefix, os.path.splitext(file)[0] + '.node')
                gpu_files.append((inp, out))
        test_files.append(gpu_files)

    print(f'Starting test on 8 GPUs, 2 images per GPU (Total 16 images)...')
    processes = []
    for i in range(8):
        if i < len(test_files) and test_files[i]:
            p = multiprocessing.Process(
                target=run_ingestion_on_gpu, 
                args=(i, test_files[i], dataset_dir, output_prefix, 2)
            )
            p.start()
            processes.append(p)
            time.sleep(2) # 모델 로딩 간격

    for p in processes:
        p.join()
        
    print('Multi-GPU test finished.')
    
    # 결과 확인
    nodes = [f for f in os.listdir(os.path.join(dataset_dir, output_prefix)) if f.endswith('.node')]
    print(f'Generated .node files: {len(nodes)}')
