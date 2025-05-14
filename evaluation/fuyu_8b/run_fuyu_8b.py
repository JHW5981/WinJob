from transformers import FuyuProcessor, FuyuForCausalLM

def run_fuyu_8b(sample, model, processor, max_new_tokens=128):
    dtype = model.dtype
    device = model.device
    image = sample['image']
    question = sample['question']
    answer = sample['answer']

    inputs = processor(text=question+'\n', images=image, return_tensors="pt").to(device=device, dtype=dtype)

    # autoregressively generate text
    generation_output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = processor.batch_decode(generation_output[:, -max_new_tokens:], skip_special_tokens=True)
    return response, answer

if __name__ == "__main__":
    import sys
    sys.path.append("/home/jihuawei2/projects/WinJob")

    from dataset.benchmarks.load_dataset import load_dataset
    from tqdm import tqdm
    import os
    import json

    processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
    model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b", device_map="auto").eval()

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
            output_text, answer = run_fuyu_8b(sample, model, processor)
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



