import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.core import SimpleDirectoryReader

from vl_embedding import VL_Embedding

class Ingestion:
    def __init__(self, dataset_dir,input_prefix='img',output_prefix='colqwen_ingestion',embed_model_name='vidore/colqwen2-v1.0', device=None):
        self.dataset_dir = dataset_dir
        self.input_dir  = os.path.join(dataset_dir, input_prefix)
        self.output_dir = os.path.join(dataset_dir, output_prefix)
        self.workers = 5
        self.embed_model_name = embed_model_name
        self.reader = SimpleDirectoryReader(input_dir = self.input_dir)
        self.pipeline = IngestionPipeline(transformations=[
            SimpleFileNodeParser(),
            VL_Embedding(model=embed_model_name,mode='image', device=device)
        ])



    def ingestion_example(self, input_file, output_file):
        documents = self.reader.load_file(Path(input_file),self.reader.file_metadata,self.reader.file_extractor)
        nodes = self.pipeline.run(documents=documents,num_workers=1, show_progress=False)
        nodes_json = [node.to_dict() for node in nodes]
        with open(output_file, 'w') as json_file:
            json.dump(nodes_json, json_file, indent=2, ensure_ascii=False)
        return True

    def ingestion_batch(self, file_pairs):
        """
        여러 개의 이미지를 한 번에 배치로 처리하여 GPU 효율을 극대화합니다.
        file_pairs: [(input_path, output_path), ...] 리스트
        """
        documents = []
        for input_path, _ in file_pairs:
            try:
                docs = self.reader.load_file(Path(input_path), self.reader.file_metadata, self.reader.file_extractor)
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {input_path}: {e}")
        
        if not documents:
            return False

        # VL_Embedding.__call__ 은 내부적으로 batch 처리를 수행함
        nodes = self.pipeline.run(documents=documents, show_progress=False)
        
        for node in nodes:
            file_name = node.metadata.get('file_name')
            if not file_name:
                continue
            file_prefix, _ = os.path.splitext(file_name)
            out_path = os.path.join(self.output_dir, file_prefix) + '.node'
            with open(out_path, 'w') as f:
                json.dump([node.to_dict()], f, indent=2, ensure_ascii=False)
        return True
    
    def ingestion_multi_session(self):
        os.makedirs(self.output_dir, exist_ok=True)
        file_to_process = []
        for file in os.listdir(self.input_dir):
            file_prefix,_ = os.path.splitext(file)
            input_file = os.path.join(self.input_dir, file)
            output_file = os.path.join(self.output_dir, file_prefix) + '.node'
            if not os.path.exists(output_file):
                file_to_process.append((input_file, output_file))
        if self.workers == 1:
            for input_file, output_file in tqdm(file_to_process):
                self.ingestion_example(input_file, output_file)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_file = {executor.submit(self.ingestion_example, input_file, output_file): (input_file, output_file) for input_file, output_file in file_to_process}
                for future in tqdm(as_completed(future_to_file), total=len(file_to_process), desc='Processing files'):
                    result_type = future.result()
    


if __name__ == '__main__':
    dataset_dir = './search_engine/corpus'
    ingestion = Ingestion(dataset_dir,input_prefix='img',output_prefix='colqwen_ingestion',embed_model_name='vidore/colqwen2-v1.0') # colqwen2
    ingestion.ingestion_multi_session()
