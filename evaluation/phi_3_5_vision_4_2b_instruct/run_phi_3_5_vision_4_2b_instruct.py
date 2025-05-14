from transformers import AutoModelForCausalLM, AutoProcessor

def run_phi_3_5_vision_4_2b_instruct(samples, model, processor):
    dtype = model.dtype
    device = model.device
    question = samples["question"]
    image = samples["image"]
    answer = samples["answer"]
    
    image = [image]
    placeholder = "<|image_1|>\n"
    
    messages = [
        {"role": "user", 
         "content": placeholder + question},
    ]
    prompt = processor.tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
    )
    
    inputs = processor(prompt, image, return_tensors="pt").to(device)
    generation_args = { 
        "max_new_tokens": 1000, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 

    generate_ids = model.generate(**inputs, 
    eos_token_id=processor.tokenizer.eos_token_id, 
    **generation_args
    )

    # remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=False)[0]
    
    return response, answer

if __name__ == "__main__":
    import sys
    sys.path.append("/home/jihuawei2/projects/WinJob")

    from dataset.benchmarks.load_dataset import load_dataset
    from tqdm import tqdm
    import os
    import json

    model_id = "microsoft/Phi-3.5-vision-instruct" 
    # Note: set _attn_implementation='eager' if you don't have flash_attn installed
    model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    trust_remote_code=True, 
    torch_dtype="auto", 
    _attn_implementation='flash_attention_2'    
    ).eval()

    # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
    processor = AutoProcessor.from_pretrained(model_id, 
    trust_remote_code=True, 
    num_crops=4
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
            output_text, answer = run_phi_3_5_vision_4_2b_instruct(sample, model, processor)
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



