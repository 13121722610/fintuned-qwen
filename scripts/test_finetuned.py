import torch
import os

# ========== 设置镜像源（最重要！） ==========
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("✅ 使用HF镜像: https://hf-mirror.com")

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from datetime import datetime

# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def test_finetuned_model():
    """测试微调后的医疗模型"""
    
    print("=" * 70)
    print("🧪 测试微调后的Qwen2.5-7B医疗模型")
    print("=" * 70)
    
    # 基础模型路径
    base_model = "Qwen/Qwen2.5-7B-Instruct"
    
    # LoRA适配器路径（你的训练输出目录）
    lora_path = "/amax/home/yhji/LM-Course/finetuned_model_20251212_090808"
    
    # 输出目录
    output_dir = "/amax/home/yhji/LM-Course/output"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📥 加载基础模型: {base_model}")
    print(f"📥 加载LoRA适配器: {lora_path}")
    
    # 1. 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True
    )
    
    # 2. 加载基础模型
    print("\n🤖 加载基础模型...")
    base_model_inst = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 3. 加载LoRA权重
    print("🎯 加载LoRA适配器...")
    model = PeftModel.from_pretrained(base_model_inst, lora_path)
    model.eval()
    
    # 4. 测试问题（与训练数据相同格式）
    test_questions = [
        "排卵日同房过后一直小腹痛腰痛怎么回事离月经期间还有九天请问医生我这是怎么了",
        "有做磁共振请医生建议，现在想问下得怎么治疗，需要用什么药物还是住院治疗动手术", 
        "大约快有一年多了，早上起来嘴里好多口水，现在早晚刷牙，牙齿也没有畸形什么的，到底是什么原因啊？",
    ]
    
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"📝 测试 {i}/{len(test_questions)}")
        print(f"💬 问题: {question}")
        
        # 构建对话（与训练时相同的格式）
        user_content = f"你是一个专业的医疗助手，请根据用户的症状描述给出专业、清晰且实用的健康建议。\n\n用户问题：{question}"
        
        messages = [
            {"role": "user", "content": user_content},
        ]
        
        # 应用Qwen的聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # 生成回答
        print("⏳ 生成回答...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取助手回答
        if "assistant" in response:
            answer = response.split("assistant")[-1].strip()
        elif user_content in response:
            answer = response.split(user_content)[-1].strip()
        else:
            answer = response
        
        print(f"💊 回答: {answer}...")
        
        results.append({
            "id": i,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
    
    # 5. 保存结果
    output_file = f"{output_dir}/finetuned_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model": "Qwen2.5-7B-Instruct-Finetuned-Medical",
            "base_model": base_model,
            "lora_path": lora_path,
            "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ 测试完成！")
    print(f"📊 共测试 {len(results)} 个问题")
    print(f"📁 结果文件: {output_file}")
    
    # 6. 显示对比（可以对比原始模型和微调后模型）
    print(f"\n📋 快速查看结果:")
    for i, result in enumerate(results[:3], 1):
        print(f"\n--- 问题 {i} ---")
        print(f"❓: {result['question']}...")
        print(f"💊: {result['answer']}...")

if __name__ == "__main__":
    test_finetuned_model()
