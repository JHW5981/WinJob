from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

def run_instructblip_vicuna_7b(sample, model, processor):
    dtype = model.dtype
    device = model.device
    image = sample['image'].convert("RGB")
    question = sample['question']
    answer = sample['answer']
    
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    
    outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
    )
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return response, answer

if __name__ == "__main__":
    import sys
    sys.path.append("/home/jihuawei2/projects/WinJob")

    from dataset.benchmarks.load_dataset import load_dataset
    from tqdm import tqdm
    import os
    import json

    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
    
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
            output_text, answer = run_instructblip_vicuna_7b(sample, model, processor)
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

