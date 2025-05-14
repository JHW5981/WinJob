from utils import load_image


def run_internvl3_2b_instruct(sample, model, tokenizer):
    device = model.device
    dtype = model.dtype
    image = sample['image']
    question = sample['question']
    answer = sample['answer']

    pixel_values = load_image(image, max_num=12).to(device=device, dtype=dtype)
    generation_config = dict(max_new_tokens=1024, do_sample=True)

    # single-image single-round conversation (单图单轮对话)
    question = f"<image>\n{question}"
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response, answer

if __name__ == "__main__":
    import sys
    sys.path.append("/home/jihuawei2/projects/WinJob")

    import os
    import json
    import torch
    from tqdm import tqdm
    from dataset.benchmarks.load_dataset import load_dataset
    from transformers import AutoTokenizer, AutoModel

    path = "OpenGVLab/InternVL3-2B"
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True).eval().to("cuda:1")
    tokenizer = AutoTokenizer.from_pretrained(
        path, 
        trust_remote_code=True, 
        use_fast=False,
        pad_token_id=151645  # 显式设置pad_token_id
    )
    
    datasets = {
        "aokvqa": load_dataset("aokvqa", "test"),
        "chartvqa": load_dataset("chartvqa", "test"), 
        "docvqa": load_dataset("docvqa", "val"),
        "infovqa": load_dataset("infovqa", "val"),
        "tablevqa": load_dataset("tablevqa", "test"),
        "textvqa": load_dataset("textvqa", "val")
    }

    for dataset_name, dataset in datasets.items():
        for sample in tqdm(dataset, desc=f"Processing {dataset_name}"):
            output_text, answer = run_internvl3_2b_instruct(sample, model, tokenizer)
            question = sample['question']
            answer = sample['answer']
            prediction = output_text
            
            # save to json
            current_file = os.path.abspath(__file__)
            parent_dir = os.path.dirname(current_file)
            
            result_file = os.path.join(parent_dir, f"{dataset_name}_test.json")
            if os.path.exists(result_file):
                with open(result_file, "r") as f:
                    results = json.load(f)
            else:
                results = []
                
            results.append({
                "question": question,
                "answer": answer,
                "prediction": prediction
            })
            
            with open(result_file, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)











