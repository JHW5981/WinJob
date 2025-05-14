from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import os

def run_qwen_2_5_7b_instruct(sample, model, processor, max_new_tokens=128):
    device = model.device
    image = sample['image']
    question = sample['question']
    answer = sample['answer']

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0], answer

if __name__ == "__main__":
    import sys
    sys.path.append("/home/jihuawei2/projects/WinJob")

    from tqdm import tqdm
    from dataset.benchmarks.load_dataset import load_dataset

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        ).eval()
    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    datasets = {
        # "aokvqa": load_dataset("aokvqa", "test"),
        # "chartvqa": load_dataset("chartvqa", "test"), 
        # "docvqa": load_dataset("docvqa", "val"),
        # "infovqa": load_dataset("infovqa", "val"),
        "tablevqa": load_dataset("tablevqa", "test"),
        "textvqa": load_dataset("textvqa", "val")
    }

    for dataset_name, dataset in datasets.items():
        for sample in tqdm(dataset, desc=f"Processing {dataset_name}"):
            output_text, answer = run_qwen_2_5_7b_instruct(sample, model, processor)
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
    
    

