import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

def run_llava_1_5_7b(sample, model, processor, max_new_tokens=128):
    device = model.device
    dtype = model.dtype
    image = sample['image'].convert('RGB')
    question = sample['question']
    answer = sample['answer']

    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image"}, 
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors='pt').to(device=device, dtype=dtype)

    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    response = processor.decode(output[0][2:], skip_special_tokens=True)
    return response, answer


if __name__ == "__main__":
    import sys
    sys.path.append("/home/jihuawei2/projects/WinJob")

    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf", 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to("cuda:3")

    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    import json
    import os
    from tqdm import tqdm
    from dataset.benchmarks.load_dataset import load_dataset

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
            output_text, answer = run_llava_1_5_7b(sample, model, processor)
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



