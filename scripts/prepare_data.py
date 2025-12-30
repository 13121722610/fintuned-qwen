import json
import os
from datasets import Dataset
from datetime import datetime

def prepare_medical_dataset_correct():
    """正确的医疗数据集预处理"""
    
    # 1. 文件路径
    input_file = "/amax/home/yhji/LM-Course//data/data-10k.json"  # 修改为您的文件路径
    output_dir = "/amax/home/yhji/LM-Course/processed_data"
    
    print(f"📂 读取数据: {input_file}")
    
    # 2. ✅ 修改：读取JSON数组格式（您的格式）
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 直接加载JSON数组
    
    print(f"📊 原始数据条数: {len(data)}")
    
    # 3. 转换格式（保持不变）
    processed_data = []
    for i, item in enumerate(data):
        # 提取各部分
        system_prompt = item['instruction']  # 系统提示
        user_question = item['input']        # 用户问题
        assistant_answer = item['output']    # 助手回答
        
        # 构建正确的对话格式
        processed_data.append({
            "id": i,
            "conversations": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": assistant_answer}
            ],
            "system_prompt": system_prompt,
            "user_question": user_question,
            "assistant_answer": assistant_answer
        })
    
    # 4. 创建数据集（保持不变）
    dataset = Dataset.from_list(processed_data)
    
    # 5. 分割数据集（保持不变）
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
    
    # 6. 保存数据集（保持不变）
    os.makedirs(output_dir, exist_ok=True)
    split_dataset.save_to_disk(output_dir)
    
    # 7. 保存统计信息（保持不变）
    stats = {
        "total_samples": len(data),
        "train_samples": len(split_dataset['train']),
        "val_samples": len(split_dataset['test']),
        "split_ratio": "90%训练, 10%验证",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_format": "system_prompt + user_question + assistant_answer",
        "format_example": {
            "system": "你是一个医疗问诊专家，请根据用户的问题给出专业的回答",
            "user": "具体问题...",
            "assistant": "专业回答..."
        }
    }
    
    stats_file = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 数据预处理完成！")
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 训练集: {len(split_dataset['train'])} 条")
    print(f"📊 验证集: {len(split_dataset['test'])} 条")
    print(f"📊 统计文件: {stats_file}")
    
    # 8. 显示样本示例
    print(f"\n📝 前3条样本示例:")
    for i in range(min(3, len(processed_data))):
        print(f"\n--- 样本 {i+1} ---")
        print(f"系统: {processed_data[i]['conversations'][0]['content'][:50]}...")
        print(f"用户: {processed_data[i]['conversations'][1]['content'][:50]}...")
        print(f"助手: {processed_data[i]['conversations'][2]['content'][:50]}...")
    
    return split_dataset

if __name__ == "__main__":
    prepare_medical_dataset_correct()