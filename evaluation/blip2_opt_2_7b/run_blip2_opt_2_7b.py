from transformers import Blip2Processor, Blip2ForConditionalGeneration

def run_blip2_opt_2_7b(sample, model, processor):
    device = model.device
    dtype = model.dtype
    image = sample['image'].convert('RGB')
    question = sample['question']
    answer = sample['answer']

    inputs = processor(image, question, return_tensors="pt").to(device=device, dtype=dtype)
    out = model.generate(**inputs)
    response = processor.decode(out[0], skip_special_tokens=True).strip()
    return response, answer
    

if __name__ == "__main__":
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")

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
            output_text, answer = run_blip2_opt_2_7b(sample, model, processor)
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







