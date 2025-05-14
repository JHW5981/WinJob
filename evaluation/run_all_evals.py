import os
import json
from eval_utils import evaluate

def run_all_evaluations():
    base_path = "/home/jihuawei2/projects/WinJob/evaluation"
    results = {}
    
    # 获取所有模型文件夹
    model_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for model_dir in model_dirs:
        model_path = os.path.join(base_path, model_dir)
        results[model_dir] = {}
        
        # 获取该模型文件夹下的所有json文件
        json_files = [f for f in os.listdir(model_path) if f.endswith('.json')]
        
        for json_file in json_files:
            try:
                with open(os.path.join(model_path, json_file), 'r') as f:
                    samples = json.load(f)
                
                # 计算评估指标
                metrics = evaluate(samples)
                results[model_dir][json_file] = metrics
                
                print(f"已评估 {model_dir} - {json_file}: {metrics}")
            except Exception as e:
                print(f"处理 {model_dir}/{json_file} 时出错: {str(e)}")
    
    # 保存所有结果
    output_path = os.path.join(base_path, 'evaluation_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results

if __name__ == "__main__":
    results = run_all_evaluations()
    print("\n=== 最终结果 ===")
    for model, model_results in results.items():
        print(f"\n{model}:")
        for dataset, metrics in model_results.items():
            print(f"  {dataset}: {metrics}")
