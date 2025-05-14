# Load benchmarks
import pandas as pd
import os
from PIL import Image
import io
import random
import numpy as np

class VQADataset:
    def __init__(self, dataset_df):
        if isinstance(dataset_df, pd.DataFrame):
            self.dataset_df = dataset_df.to_dict('records')
        else:
            self.dataset_df = dataset_df

    def __len__(self):
        return len(self.dataset_df)
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            return VQADataset(self.dataset_df[index])
        
        sample = self.dataset_df[index]
        image_binary = sample['image']['bytes']
        image = Image.open(io.BytesIO(image_binary))
        if 'question' in sample.keys():
            question = sample['question']
        elif 'query' in sample.keys():
            question = sample['query']
        else:
            raise ValueError(f"No question found in sample {sample}")
        if 'direct_answers' in sample.keys() and sample['direct_answers'] is not None:
            answer = sample['direct_answers']
        elif 'choices' in sample.keys():
            answer = sample['choices']
        elif 'answer' in sample.keys():
            answer = sample['answer']
        elif 'answers' in sample.keys():
            answer = sample['answers']
        elif 'gt' in sample.keys():
            answer = sample['gt']
        elif 'label' in sample.keys():
            answer = sample['label']
        else:
            raise ValueError(f"No answer found in sample {sample}")
        
        # 处理不同类型的答案
        if isinstance(answer, str):
            try:
                answer = eval(answer)
            except:
                answer = [answer]
        elif isinstance(answer, np.ndarray):  # 处理numpy数组
            answer = answer.tolist()
        elif not isinstance(answer, (list, tuple)):
            answer = [answer]

        answer = random.choice(answer) if isinstance(answer, list) else answer
        return {
            'image': image,
            'question': question,
            'answer': answer
        }

def load_dataset(dataset_name, split):
    """
    按需加载指定数据集的指定split
    
    Args:
        dataset_name: 数据集名称,可选 'aokvqa', 'chartvqa', 'docvqa', 'infovqa', 'tablevqa', 'textvqa'
        split: 数据集分割,可选 'train', 'val', 'test'
    
    Returns:
        VQADataset对象
    """
    base_path = '/home/jihuawei2/Benchmarks'
    
    if dataset_name == 'aokvqa':
        if split == 'train':
            df1 = pd.read_parquet(f'{base_path}/a-okvqa/data/train-00000-of-00002-c1d24de3bacb5e0c.parquet')
            df2 = pd.read_parquet(f'{base_path}/a-okvqa/data/train-00001-of-00002-6b4f3abe2dc385d0.parquet')
            df = pd.concat([df1, df2], ignore_index=True)
        elif split == 'val':
            df = pd.read_parquet(f'{base_path}/a-okvqa/data/validation-00000-of-00001-b2bd0de231b6326a.parquet')
        else:
            df = pd.read_parquet(f'{base_path}/a-okvqa/data/test-00000-of-00001-d306bf3ad53b6618.parquet')
            
    elif dataset_name == 'chartvqa':
        if split == 'train':
            df = pd.DataFrame()
            for i in os.listdir(f"{base_path}/chartvqa/data"):
                if i.startswith("train"):
                    df = pd.concat([df, pd.read_parquet(f"{base_path}/chartvqa/data/{i}")])
            df = df.reset_index(drop=True)
        elif split == 'val':
            df = pd.read_parquet(f'{base_path}/chartvqa/data/val-00000-of-00001-0f11003c77497969.parquet')
        else:
            df = pd.read_parquet(f'{base_path}/chartvqa/data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet')
            
    elif dataset_name == 'docvqa':
        df = pd.DataFrame()
        for i in os.listdir(f"{base_path}/docvqa/DocVQA"):
            if i.startswith(split):
                df = pd.concat([df, pd.read_parquet(f"{base_path}/docvqa/DocVQA/{i}")])
        df = df.reset_index(drop=True)
        
    elif dataset_name == 'infovqa':
        df = pd.DataFrame()
        for i in os.listdir(f"{base_path}/infovqa/InfographicVQA"):
            if i.startswith(split):
                df = pd.concat([df, pd.read_parquet(f"{base_path}/infovqa/InfographicVQA/{i}")])
        df = df.reset_index(drop=True)
        
    elif dataset_name == 'tablevqa':
        # 读取所有子数据集
        datasets = {
            'fintabnetqa': pd.read_parquet(f'{base_path}/tablevqa-bench/data/fintabnetqa-00000-of-00001-c337fe9eb7a70460.parquet'),
            'vtabfactqa': pd.read_parquet(f'{base_path}/tablevqa-bench/data/vtabfact-00000-of-00001-ecd1dbae37761ddd.parquet'),
            'vwtq_synqa': pd.read_parquet(f'{base_path}/tablevqa-bench/data/vwtq_syn-00000-of-00001-2daaa7285aca2c1d.parquet'),
            'vwtqqa': pd.read_parquet(f'{base_path}/tablevqa-bench/data/vwtq-00000-of-00001-764eb826ab450a91.parquet')
        }
        
        # 对每个子数据集进行分割并合并
        split_dfs = []
        for dataset in datasets.values():
            if split == 'train':
                split_df = dataset.iloc[:int(len(dataset)*0.8)]
            elif split == 'val':
                split_df = dataset.iloc[int(len(dataset)*0.8):int(len(dataset)*0.9)]
            else:
                split_df = dataset.iloc[int(len(dataset)*0.9):]
            split_dfs.append(split_df)
        df = pd.concat(split_dfs)
        
    elif dataset_name == 'textvqa':
        df = pd.DataFrame()
        for i in os.listdir(f"{base_path}/textvqa/data"):
            if i.startswith(split):
                df = pd.concat([df, pd.read_parquet(f"{base_path}/textvqa/data/{i}")])
        df = df.reset_index(drop=True)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    return VQADataset(df)

if __name__ == "__main__":
    # 测试代码
    datasets = {
        "aokvqa": load_dataset("aokvqa", "test"),
        "chartvqa": load_dataset("chartvqa", "test"), 
        "docvqa": load_dataset("docvqa", "val"),
        "infovqa": load_dataset("infovqa", "val"),
        "tablevqa": load_dataset("tablevqa", "test"),
        "textvqa": load_dataset("textvqa", "val")
    }
    
    for name, dataset in datasets.items():
        print(f"\n检查 {name} 数据集:")
        print(f"数据集大小: {len(dataset)}")
        # 检查第一个样本
        sample = dataset[0]
        print(f"第一个样本的键: {sample.keys()}")
        print(f"答案字段: {sample.get('answer', sample.get('answers', sample.get('gt', sample.get('label', None))))}")
