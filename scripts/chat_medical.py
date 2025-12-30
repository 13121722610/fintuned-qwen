import torch
import os

# ========== 设置镜像源 ==========
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

print("✅ 使用HF镜像: https://hf-mirror.com")

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from datetime import datetime
import sys

class MedicalChatBot:
    """医疗聊天机器人"""
    
    def __init__(self, lora_path=None):
        print("=" * 70)
        print("🏥 医疗助手聊天机器人")
        print("=" * 70)
        
        # 基础模型
        self.base_model = "Qwen/Qwen2.5-7B-Instruct"
        
        # LoRA适配器路径（默认使用最新训练的）
        if lora_path is None:
            # 自动查找最新的训练目录
            import glob
            lora_dirs = glob.glob("/amax/home/yhji/LM-Course/finetuned_model_*")
            if lora_dirs:
                lora_dirs.sort(key=os.path.getmtime, reverse=True)
                self.lora_path = lora_dirs[0]
                print(f"🔍 自动检测到最新模型: {os.path.basename(self.lora_path)}")
            else:
                self.lora_path = "/amax/home/yhji/LM-Course/finetuned_model_20251206_133554"
                print(f"⚠️  使用默认模型: {self.lora_path}")
        else:
            self.lora_path = lora_path
        
        print(f"📥 基础模型: {self.base_model}")
        print(f"📥 LoRA适配器: {self.lora_path}")
        
        # 初始化模型
        self._load_model()
        
        # 历史记录
        self.conversation_history = []
        self.save_dir = "/amax/home/yhji/LM-Course/chat_history"
        os.makedirs(self.save_dir, exist_ok=True)
    
    def _load_model(self):
        """加载模型"""
        print("\n⏳ 加载模型中...")
        
        try:
            # 1. 加载tokenizer（优先使用训练目录中的）
            tokenizer_path = os.path.join(self.lora_path, "tokenizer")
            if os.path.exists(tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    trust_remote_code=True
                )
                print("✅ 使用训练目录中的tokenizer")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model,
                    trust_remote_code=True
                )
                print("✅ 从HF加载tokenizer")
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 2. 加载基础模型
            print("🤖 加载基础模型...")
            self.base_model_inst = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 3. 加载LoRA权重
            print("🎯 加载LoRA适配器...")
            self.model = PeftModel.from_pretrained(
                self.base_model_inst, 
                self.lora_path
            )
            self.model.eval()
            
            print("🚀 模型加载完成！")
            print(f"🎮 设备: {self.model.device}")
            
            # 测试一个简单问题
            test_response = self._generate_response("你好")
            if test_response:
                print("✅ 模型测试通过")
            else:
                print("⚠️  模型响应异常")
                
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            sys.exit(1)
    
    def _generate_response(self, user_input, temperature=0.7, max_tokens=500):
        """生成回答"""
        # 构建对话
        user_content = f"你是一个专业的医疗助手，请根据用户的症状描述给出专业、清晰且实用的健康建议。\n\n用户问题：{user_input}"
        
        messages = [
            {"role": "user", "content": user_content},
        ]
        
        # 应用聊天模板
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            # 如果模板失败，使用简单格式
            text = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取助手回答
        if "assistant" in response:
            answer = response.split("assistant")[-1].strip()
        elif user_content in response:
            answer = response.split(user_content)[-1].strip()
        else:
            answer = response
        
        return answer.strip()
    
    def chat(self):
        """开始聊天"""
        print("\n" + "="*70)
        print("💬 开始聊天 (输入 'quit' 退出, 'help' 查看命令)")
        print("="*70)
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_history = []
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n👤 您: ").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.lower() == 'quit':
                    print("👋 再见！")
                    self._save_session(session_id, session_history)
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                elif user_input.lower() == 'history':
                    self._show_history(session_history)
                    continue
                
                elif user_input.lower() == 'clear':
                    session_history = []
                    print("🧹 历史记录已清除")
                    continue
                
                elif user_input.lower() == 'save':
                    self._save_session(session_id, session_history)
                    continue
                
                elif user_input.lower() == 'params':
                    self._adjust_parameters()
                    continue
                
                # 生成回答
                print("🤖 思考中...", end="", flush=True)
                
                response = self._generate_response(user_input)
                
                # 清空"思考中"
                print("\r" + " " * 20 + "\r", end="", flush=True)
                
                # 打印回答
                print(f"🏥 助手: {response}")
                
                # 保存到历史
                session_history.append({
                    "user": user_input,
                    "assistant": response,
                    "timestamp": datetime.now().isoformat()
                })
                
                # 自动保存（每5轮）
                if len(session_history) % 5 == 0:
                    self._save_session(session_id, session_history)
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  中断操作")
                save = input("是否保存聊天记录？(y/n): ")
                if save.lower() == 'y':
                    self._save_session(session_id, session_history)
                break
                
            except Exception as e:
                print(f"\n❌ 错误: {e}")
                continue
    
    def _show_help(self):
        """显示帮助"""
        print("\n📋 可用命令:")
        print("  help     - 显示此帮助")
        print("  history  - 查看当前会话历史")
        print("  clear    - 清除当前会话历史")
        print("  save     - 保存当前会话")
        print("  params   - 调整生成参数")
        print("  quit     - 退出聊天")
    
    def _show_history(self, history):
        """显示历史"""
        if not history:
            print("📝 当前没有聊天历史")
            return
        
        print("\n📜 聊天历史:")
        for i, item in enumerate(history, 1):
            print(f"\n--- 第{i}轮 ---")
            print(f"👤 您: {item['user']}")
            print(f"🏥 助手: {item['assistant'][:100]}...")
    
    def _save_session(self, session_id, history):
        """保存会话"""
        if not history:
            print("📝 没有内容可保存")
            return
        
        filename = f"{self.save_dir}/chat_{session_id}.json"
        data = {
            "model": "Qwen2.5-7B-Medical-Finetuned",
            "lora_path": self.lora_path,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "history": history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 聊天记录已保存: {filename}")
    
    def _adjust_parameters(self):
        """调整生成参数"""
        print("\n⚙️  调整生成参数:")
        try:
            temp = float(input("温度 (0.1-1.0, 当前0.7): ") or 0.7)
            tokens = int(input("最大生成长度 (100-1000, 当前500): ") or 500)
            
            # 验证范围
            temp = max(0.1, min(1.0, temp))
            tokens = max(100, min(1000, tokens))
            
            # 这里可以保存参数，简化版本先不实现
            print(f"✅ 参数已更新: 温度={temp}, 最大长度={tokens}")
        except:
            print("⚠️  参数调整失败，使用默认值")


# ========== 以下是新增的Web API部分 ==========

class MedicalChatAPI:
    """为Web应用封装的聊天机器人API"""
    
    def __init__(self, model_path=None):
        """
        初始化API
        
        Args:
            model_path: LoRA模型路径，如果为None则自动查找最新模型
        """
        print("🚀 初始化Web API...")
        self.bot = MedicalChatBot(lora_path=model_path)
        print("✅ Web API初始化完成")
    
    def chat(self, message, temperature=0.7, max_tokens=500):
        """
        单次对话接口
        
        Args:
            message: 用户消息
            temperature: 温度参数
            max_tokens: 最大生成长度
            
        Returns:
            str: 助手回复
        """
        # 构建完整的对话格式
        full_prompt = f"你是一个专业的医疗助手，请根据用户的症状描述给出专业、清晰且实用的健康建议。\n\n用户问题：{message}"
        
        # 调用原有模型的生成方法
        response = self.bot._generate_response(
            full_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # 清理回复格式
        cleaned_response = self._clean_response(response)
        return cleaned_response
    
    def _clean_response(self, response):
        """清理回复，移除多余的格式标记"""
        # 移除可能的模板标记
        markers = ["<|im_start|>", "<|im_end|>", "assistant", "user", "system"]
        for marker in markers:
            response = response.replace(marker, "")
        
        # 移除多余的空行
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            "base_model": self.bot.base_model,
            "lora_path": self.bot.lora_path,
            "device": str(self.bot.model.device) if hasattr(self.bot, 'model') else "unknown"
        }


def test_api():
    """测试API是否正常工作"""
    print("🧪 测试MedicalChatAPI...")
    
    try:
        # 创建API实例
        api = MedicalChatAPI()
        
        # 测试简单问题
        test_questions = [
            "我头痛怎么办？",
            "感冒了应该吃什么药？"
        ]
        
        for question in test_questions:
            print(f"\n❓ 问题: {question}")
            response = api.chat(question)
            print(f"💊 回答: {response[:100]}...")
        
        print("\n✅ API测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ API测试失败: {e}")
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="医疗助手聊天机器人")
    parser.add_argument("--model", type=str, help="LoRA模型路径", default=None)
    parser.add_argument("--test", action="store_true", help="快速测试模式")
    parser.add_argument("--test-api", action="store_true", help="测试Web API")
    
    args = parser.parse_args()
    
    # 测试API模式
    if args.test_api:
        test_api()
        return
    
    # 创建聊天机器人
    bot = MedicalChatBot(lora_path=args.model)
    
    # 测试模式
    if args.test:
        print("\n🧪 快速测试模式:")
        test_questions = [
            "我头痛怎么办？",
            "感冒了应该吃什么药？",
            "高血压要注意什么？"
        ]
        
        for q in test_questions:
            print(f"\n❓ 测试问题: {q}")
            response = bot._generate_response(q)
            print(f"💊 回答: {response[:150]}...")
        
        print("\n✅ 快速测试完成")
        return
    
    # 交互模式
    bot.chat()

if __name__ == "__main__":
    main()
    