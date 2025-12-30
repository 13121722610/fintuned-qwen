import torch
import os

# ========== 设置镜像源（最重要！） ==========
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("✅ 使用HF镜像: https://hf-mirror.com")

from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime

def main():
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    
    # 1. 设置输出目录和文件路径
    OUTPUT_DIR = "/amax/home/yhji/LM-Course/output"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "qwen_baseline_responses.json")
    
    # 2. 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"🚀 开始测试: {MODEL_NAME}")
    print(f"🎮 可用GPU数量: {torch.cuda.device_count()}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"💾 输出文件: {OUTPUT_FILE}")
    
    # 3. 测试问题
    test_questions = [
        "排卵日同房过后一直小腹痛腰痛怎么回事离月经期间还有九天请问医生我这是怎么了",
        "你好，全身没劲，没精神，吃不下饭，只想睡觉，是什么情况",
        "现年34岁，医生诊断是眼睛里血管堵塞，请问怎样能治好谢谢",
    ]
    
    print("⏳ 正在加载模型...")
    
    try:
        # 4. 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        
        # 5. 加载模型 - 使用多GPU
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"✅ 模型加载完成！")
        if hasattr(model, 'hf_device_map'):
            print(f"📊 模型分布在以下设备: {model.hf_device_map}")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("尝试使用CPU模式...")
        
        # 备用方案：使用CPU
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        print("✅ 模型加载完成（CPU模式）")
    
    model.eval()
    
    # 6. 生成回答
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"📝 测试 {i}/{len(test_questions)}: {question}")
        
        # 构建对话
        messages = [
            {"role": "system", "content": "你是一个专业的医疗助手。"},
            {"role": "user", "content": question}
        ]
        
        # 格式化
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码
        device = model.device if hasattr(model, 'device') else 'cpu'
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=350,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取助手回答
        if "assistant" in response:
            answer = response.split("assistant")[-1].strip()
        else:
            # 备选提取方法
            answer = response.split(question)[-1].strip() if question in response else response
        
        print(f"💬 模型回答: {answer}")
        
        results.append({
            "id": i,
            "question": question,
            "response": answer,
            "model": MODEL_NAME,
            "timestamp": datetime.now().isoformat(),
            "device": str(device)
        })
    
    # 7. 保存结果
    output_data = {
        "model": MODEL_NAME,
        "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_info": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "results": results
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ 测试完成！")
    print(f"📊 共测试 {len(results)} 个问题")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"💾 结果文件: {os.path.basename(OUTPUT_FILE)}")
    
    # 8. 显示目录内容
    print(f"\n📂 输出目录内容:")
    try:
        files = os.listdir(OUTPUT_DIR)
        for file in files:
            file_path = os.path.join(OUTPUT_DIR, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path) / 1024  # KB
                print(f"  {file} ({size:.1f} KB)")
    except Exception as e:
        print(f"  无法列出目录内容: {e}")
    
    # 9. 显示GPU使用情况（如果可用）
    if torch.cuda.is_available():
        print(f"\n🎮 GPU使用统计:")
        for i in range(torch.cuda.device_count()):
            mem_used = torch.cuda.memory_allocated(i) / 1024**3
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {mem_used:.1f}/{mem_total:.1f} GB ({mem_used/mem_total*100:.1f}%)")

if __name__ == "__main__":
    main()
    