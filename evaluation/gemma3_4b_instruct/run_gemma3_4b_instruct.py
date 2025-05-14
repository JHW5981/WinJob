import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

def run_gemma3_4b_instruct(sample, model, processor, max_new_tokens=128):
    dtype = model.dtype
    device = model.device
    image = sample['image']
    question = sample['question']
    answer = sample['answer']

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(device=device, dtype=dtype)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generation = generation[0][input_len:]
    
    response = processor.decode(generation, skip_special_tokens=True)
    return response, answer

if __name__ == "__main__":
    import sys
    sys.path.append("/home/jihuawei2/projects/WinJob")

    from dataset.benchmarks.load_dataset import load_dataset
    from tqdm import tqdm
    import os
    import json

    model = Gemma3ForConditionalGeneration.from_pretrained(
        "google/gemma-3-4b-it", device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

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
            output_text, answer = run_gemma3_4b_instruct(sample, model, processor)
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







    
