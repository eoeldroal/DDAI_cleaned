import os
import time
from search_engine.ingestion import Ingestion

if __name__ == '__main__':
    dataset_dir = './search_engine/corpus'
    # 테스트를 위해 1개만 처리하도록 ingestion_example 직접 호출
    ingestion = Ingestion(dataset_dir, input_prefix='img', output_prefix='test_ingestion')
    
    # 첫 번째 이미지 찾기
    img_list = os.listdir('./search_engine/corpus/img')
    if not img_list:
        print("No images found")
        exit()
    
    test_img = os.path.join('./search_engine/corpus/img', img_list[0])
    test_output = './search_engine/corpus/test_ingestion/test.node'
    os.makedirs('./search_engine/corpus/test_ingestion', exist_ok=True)
    
    print(f"Testing ingestion with: {test_img}")
    start_time = time.time()
    ingestion.ingestion_example(test_img, test_output)
    end_time = time.time()
    
    print(f"Time taken for 1 image: {end_time - start_time:.2f} seconds")
