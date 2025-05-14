from transformers import AutoModelForCausalLM, AutoProcessor

def run_kimi_vl_a3b_instruct(sample, model, processor, max_new_tokens=128):
    dtype = model.dtype
    device = model.device
    image = sample['image']
    question = sample['question']
    answer = sample['answer']

    messages = [
        {"role": "user", 
         "content": [{"type": "image", "image": image}, 
                     {"type": "text", "text": question}]}
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response, answer

if __name__ == "__main__":
    import sys
    sys.path.append("/home/jihuawei2/projects/WinJob")

    from dataset.benchmarks.load_dataset import load_dataset
    from tqdm import tqdm
    import os
    import json

    model = AutoModelForCausalLM.from_pretrained(
        "moonshotai/Kimi-VL-A3B-Instruct",
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained("moonshotai/Kimi-VL-A3B-Instruct", trust_remote_code=True)


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
            output_text, answer = run_kimi_vl_a3b_instruct(sample, model, processor)
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



