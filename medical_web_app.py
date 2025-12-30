# medical_web_app.py - 简化侧边栏版本
import streamlit as st
import torch
import os
import sys
import json
import re
import time
from datetime import datetime
from pathlib import Path
import glob

# ========== 环境设置 ==========
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 项目根目录
PROJECT_ROOT = "/amax/home/yhji/LM-Course"

# ========== 导入相关库 ==========
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel, PeftConfig
    print("✅ 模型库导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    st.error(f"❌ 依赖库导入失败: {e}")
    st.stop()

# ========== Streamlit页面配置 ==========
st.set_page_config(
    page_title="🏥 医疗智能助手 - Qwen2.5微调版",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': """
        ### 医疗智能助手 v1.0
        
        **模型信息:**
        - 基础模型: Qwen2.5-7B-Instruct
        - 微调方法: LoRA (医疗领域)
        - 训练数据: 医疗问答对
        
        **功能:**
        - 专业医疗咨询
        - 症状分析建议
        - 健康指导
        
        **免责声明:** 本助手提供的信息仅供参考，不能替代专业医疗建议。
        """
    }
)

# ========== 自定义CSS样式 ==========
st.markdown("""
<style>
    /* 主标题 */
    .main-title {
        text-align: center;
        color: #1a237e;
        padding: 20px;
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 30px;
        border-left: 5px solid #1e88e5;
    }
    
    /* 聊天消息样式 */
    .user-message {
        background-color: #e3f2fd;
        padding: 15px 20px;
        border-radius: 18px 18px 4px 18px;
        margin: 10px 0 10px auto;
        max-width: 85%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border: 1px solid #bbdefb;
        line-height: 1.6;
    }
    
    .assistant-message {
        background-color: #f8f9fa;
        padding: 20px 25px;
        border-radius: 18px 18px 18px 4px;
        margin: 10px auto 10px 0;
        max-width: 85%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        border-left: 4px solid #4caf50;
        line-height: 1.8;
    }
    
    /* 医疗报告特殊样式 */
    .medical-report {
        font-family: 'SimSun', 'NSimSun', serif;
        color: #333;
    }
    
    .section-title {
        color: #1a237e;
        font-weight: 700;
        font-size: 1.2em;
        margin: 20px 0 10px 0;
        padding-bottom: 5px;
        border-bottom: 2px solid #1e88e5;
    }
    
    .subsection-title {
        color: #0d47a1;
        font-weight: 600;
        font-size: 1.1em;
        margin: 15px 0 8px 0;
    }
    
    .content-text {
        color: #424242;
        margin-left: 15px;
        margin-bottom: 12px;
    }
    
    /* 状态指示器 */
    .status-online {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background-color: #4caf50;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .status-offline {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background-color: #f44336;
        margin-right: 8px;
    }
    
    /* 按钮样式 */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* 输入框 */
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        padding: 15px;
        font-size: 16px;
        line-height: 1.5;
    }
    
    /* 聊天容器 */
    .chat-container {
        height: 68vh;
        overflow-y: auto;
        padding: 20px;
        background-color: #fafafa;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 20px;
        scroll-behavior: smooth;
    }
    
    /* 滚动条样式 */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #bdbdbd;
        border-radius: 4px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #9e9e9e;
    }
    
    /* 免责声明 */
    .disclaimer-box {
        background-color: #fff3cd;
        padding: 15px 20px;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin-top: 20px;
        font-size: 0.9em;
    }
    
    /* 小标题 */
    .sub-heading {
        color: #546e7a;
        font-size: 0.95em;
        margin-top: 5px;
    }
    
    /* 简化版侧边栏样式 */
    .simple-sidebar {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ========== 模型加载类 ==========
class MedicalModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_loaded = False
        self.model_info = {}
        
    def load_model(self, model_path=None):
        """加载微调后的模型"""
        try:
            st.sidebar.info("🚀 开始加载模型...")
            
            # 清空GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                st.sidebar.info("🔄 已清空GPU缓存")
            
            # 1. 自动查找最新模型
            if model_path is None:
                model_dirs = glob.glob(f"{PROJECT_ROOT}/finetuned_model_*")
                if not model_dirs:
                    st.sidebar.error("❌ 未找到训练好的模型")
                    return False
                
                model_dirs.sort(key=os.path.getmtime, reverse=True)
                model_path = model_dirs[0]
                self.model_info['path'] = os.path.basename(model_path)
                st.sidebar.success(f"📂 使用模型: {self.model_info['path']}")
            
            # 2. 配置4位量化以减少显存
            st.sidebar.info("⚙️ 配置量化加载...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            # 3. 加载tokenizer
            st.sidebar.info("🔤 加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-7B-Instruct",
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 4. 使用4位量化加载基础模型
            st.sidebar.info("🤖 加载基础模型 (4位量化)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-7B-Instruct",
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                use_cache=False
            )
            
            # 5. 加载LoRA适配器
            st.sidebar.info("🎯 加载LoRA适配器...")
            self.model = PeftModel.from_pretrained(self.model, model_path)
            self.model.eval()
            
            # 6. 设置设备
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                st.sidebar.success(f"✅ 模型加载到: {self.device}")
                
                # 显存统计
                allocated = torch.cuda.memory_allocated() / 1024**3
                self.model_info['gpu_memory'] = allocated
                st.sidebar.info(f"📊 显存占用: {allocated:.2f} GB")
            else:
                self.device = torch.device("cpu")
                st.sidebar.warning("⚠️ 使用CPU模式")
            
            self.model_loaded = True
            
            # 测试推理
            st.sidebar.info("🧪 测试模型推理...")
            test_text = "测试模型是否正常工作"
            test_input = self.tokenizer(test_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                _ = self.model.generate(**test_input, max_new_tokens=20)
            
            st.sidebar.success("✅ 模型加载完成！")
            
            return True
                
        except Exception as e:
            st.sidebar.error(f"❌ 模型加载失败: {str(e)[:200]}...")
            import traceback
            traceback.print_exc()
            return False
    
    def format_response(self, text):
        """格式化模型输出，确保正确的标题格式"""
        # 清理多余的Markdown符号
        text = re.sub(r'#{2,}', '', text)  # 移除多个#
        text = re.sub(r'\*\*\s*', '', text)  # 移除**加空格
        text = re.sub(r'\*\*([^\\*]+)\*\*', r'【\1】', text)  # 转换**内容**为【内容】
        
        # 统一标题格式
        patterns = [
            (r'^[\s]*一[、.]?\s*病情分析', '【一、病情分析】'),
            (r'^[\s]*二[、.]?\s*原因分析', '【二、原因分析】'),
            (r'^[\s]*三[、.]?\s*治病建议', '【三、治病建议】'),
            (r'^[\s]*1[、.]?\s*症状全面评估', '【1. 症状全面评估：】'),
            (r'^[\s]*2[、.]?\s*可能疾病判断', '【2. 可能疾病判断：】'),
            (r'^[\s]*1[、.]?\s*主要病因解析', '【1. 主要病因解析：】'),
            (r'^[\s]*2[、.]?\s*鉴别诊断要点', '【2. 鉴别诊断要点：】'),
            (r'^[\s]*1[、.]?\s*就医指导', '【1. 就医指导：】'),
            (r'^[\s]*2[、.]?\s*治疗方案建议', '【2. 治疗方案建议：】'),
        ]
        
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            formatted_line = line
            for pattern, replacement in patterns:
                formatted_line = re.sub(pattern, replacement, formatted_line, flags=re.IGNORECASE)
            formatted_lines.append(formatted_line)
        
        # 添加换行和缩进
        formatted_text = '\n'.join(formatted_lines)
        formatted_text = formatted_text.replace('【一、病情分析】', '\n【一、病情分析】')
        formatted_text = formatted_text.replace('【二、原因分析】', '\n\n【二、原因分析】')
        formatted_text = formatted_text.replace('【三、治病建议】', '\n\n【三、治病建议】')
        
        return formatted_text.strip()
    
    def generate_response(self, user_input, max_tokens=800):
        """生成回答"""
        if not self.model_loaded:
            return "⚠️ 模型未加载，请先加载模型"
        
        try:
            # 🛠️ 严格的格式指令
            system_prompt = """你是一个专业的医疗助手，请根据用户的症状描述给出专业、清晰且实用的健康建议。

**必须严格按照以下格式回答（不要使用任何Markdown符号如#、*、**等）：**

【一、病情分析】
【1. 症状全面评估：】
【2. 可能疾病判断：】

【二、原因分析】
【1. 主要病因解析：】
【2. 鉴别诊断要点：】

【三、治病建议】
【1. 就医指导：】
【2. 治疗方案建议：】

规则：
1. 使用【】包裹所有标题
2. 每个标题单独一行
3. 标题后换行再写具体内容
4. 内容要详细具体，每个部分至少2-3句话
5. 不要添加任何额外的格式符号
"""
            
            # 强化格式要求
            formatted_user_input = f"用户症状描述：{user_input}\n\n请严格按照上述格式要求回答，使用【】包裹标题，不要使用任何Markdown符号。"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_user_input}
            ]
            
            # 应用聊天模板
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                # 备用模板
                text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{formatted_user_input}<|im_end|>\n<|im_start|>assistant\n"
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            input_length = inputs['input_ids'].shape[1]
            
            # 生成（固定参数）
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.5,  # 固定温度
                    top_p=0.85,
                    do_sample=True,
                    repetition_penalty=1.2,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    num_beams=1,
                    no_repeat_ngram_size=3
                )
            
            # 解码
            generated_ids = outputs[0][input_length:]  # 只取新生成的部分
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 清理和格式化
            response = self.format_response(response)
            
            # 如果格式仍然有问题，进行最终清理
            response = response.replace('###', '').replace('####', '').replace('#####', '')
            response = re.sub(r'\n\s*\n\s*\n+', '\n\n', response)  # 移除多余空行
            
            return response
            
        except Exception as e:
            return f"❌ 生成回答时出错: {str(e)[:100]}"

# ========== 初始化模型 ==========
@st.cache_resource(show_spinner=False)
def init_model():
    """初始化模型（缓存）"""
    with st.spinner("🔄 正在加载医疗AI模型..."):
        model = MedicalModel()
        success = model.load_model()
        return model if success else None

# ========== 简化侧边栏配置 ==========
with st.sidebar:
    st.markdown("### ⚙️ 控制面板")
    
    # 模型状态
    st.markdown("#### 📊 模型状态")
    col1, col2 = st.columns([1, 3])
    with col1:
        if torch.cuda.is_available():
            st.markdown('<div class="status-online"></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-offline"></div>', unsafe_allow_html=True)
    with col2:
        status_text = "**在线 (GPU)**" if torch.cuda.is_available() else "**离线 (CPU)**"
        st.markdown(status_text)
    
    # 会话管理
    st.markdown("#### 💾 会话管理")
    if st.button("🔄 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # 导出对话
    if st.button("💾 导出为JSON", use_container_width=True):
        if 'messages' in st.session_state and st.session_state.messages:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_export_{timestamp}.json"
            export_dir = os.path.join(PROJECT_ROOT, "chat_exports")
            os.makedirs(export_dir, exist_ok=True)
            export_path = os.path.join(export_dir, filename)
            
            export_data = {
                "export_time": datetime.now().isoformat(),
                "total_messages": len(st.session_state.messages),
                "model": "Qwen2.5-7B-Instruct-Finetuned-Medical",
                "messages": st.session_state.messages
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            st.sidebar.success(f"✅ 已导出到:\n`{export_path}`")

# ========== 主页面 ==========
# 标题区域
st.markdown("""
<div class="main-title">
    <h1>🏥 医疗智能助手</h1>
    <p class="sub-heading">基于 Qwen2.5-7B-Instruct 微调的医疗问答系统</p>
</div>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
    # 添加欢迎消息
    welcome_msg = """您好！我是专业的医疗助手，可以为您提供健康咨询和建议。

**使用说明：**
1. 详细描述您的症状或健康问题
2. 我会按照标准医疗报告格式为您分析
3. 我的回答将包含病情分析、原因分析和治疗建议

**请注意：** 我的回答仅供参考，不能替代专业医疗诊断。"""
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": welcome_msg,
        "timestamp": datetime.now().isoformat()
    })

# 初始化模型
model = init_model()

if model is None:
    st.error("""
    ⚠️ **模型加载失败！**
    
    可能的原因：
    1. 未找到训练好的模型文件
    2. GPU显存不足
    3. 依赖库未正确安装
    
    请检查：
    - 模型文件路径是否正确
    - 确保已完成模型训练
    - 重启应用或服务器
    """)
    st.stop()

# 聊天显示区域
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>👤 您:</strong><br>
                {message['content']}
            </div>
            """, unsafe_allow_html=True)
        elif message["role"] == "assistant":
            content = message['content']
            
            # 检查是否为医疗报告格式
            if "【一、病情分析】" in content:
                # 使用医疗报告样式
                st.markdown(f"""
                <div class="assistant-message medical-report">
                    <strong>🏥 医疗报告:</strong><br><br>
                """, unsafe_allow_html=True)
                
                # 分割内容
                sections = re.split(r'【[一二三]、[^】]+】', content)
                titles = re.findall(r'【[一二三]、[^】]+】', content)
                
                # 显示每个部分
                for i, (title, section) in enumerate(zip(titles, sections[1:] if len(sections) > 1 else [content])):
                    # 处理病情分析
                    if i == 0 and "一、病情分析" in title:
                        st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
                        # 提取子部分
                        subsections = re.split(r'【[0-9]+\.[^】]+】', section)
                        subtitles = re.findall(r'【[0-9]+\.[^】]+】', section)
                        
                        for j, (subtitle, subcontent) in enumerate(zip(subtitles, subsections[1:] if len(subsections) > 1 else [section])):
                            st.markdown(f'<div class="subsection-title">{subtitle}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="content-text">{subcontent.strip()}</div>', unsafe_allow_html=True)
                    
                    # 处理原因分析
                    elif i == 1 and "二、原因分析" in title:
                        st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
                        subsections = re.split(r'【[0-9]+\.[^】]+】', section)
                        subtitles = re.findall(r'【[0-9]+\.[^】]+】', section)
                        
                        for j, (subtitle, subcontent) in enumerate(zip(subtitles, subsections[1:] if len(subsections) > 1 else [section])):
                            st.markdown(f'<div class="subsection-title">{subtitle}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="content-text">{subcontent.strip()}</div>', unsafe_allow_html=True)
                    
                    # 处理治疗建议
                    elif i == 2 and "三、治病建议" in title:
                        st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
                        subsections = re.split(r'【[0-9]+\.[^】]+】', section)
                        subtitles = re.findall(r'【[0-9]+\.[^】]+】', section)
                        
                        for j, (subtitle, subcontent) in enumerate(zip(subtitles, subsections[1:] if len(subsections) > 1 else [section])):
                            st.markdown(f'<div class="subsection-title">{subtitle}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="content-text">{subcontent.strip()}</div>', unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                # 普通消息
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>🏥 助手:</strong><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # JavaScript自动滚动到底部
    st.markdown("""
    <script>
        function scrollToBottom() {
            var container = document.querySelector('.chat-container');
            if (container) {
                container.scrollTop = container.scrollHeight;
            }
        }
        // 页面加载时滚动
        window.onload = scrollToBottom;
        // Streamlit每次更新后滚动
        setTimeout(scrollToBottom, 100);
    </script>
    """, unsafe_allow_html=True)

# 输入区域
st.markdown("### 💬 请输入您的医疗问题")

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area(
        "",
        placeholder="请详细描述您的症状、持续时间、伴随症状等（例如：我头痛、发烧3天了，还伴有咳嗽，体温最高38.5℃）",
        height=120,
        key="user_input",
        max_chars=1000,
        help="描述越详细，分析越准确"
    )
    
    col1, col2 = st.columns([1, 6])
    with col1:
        submit_button = st.form_submit_button(
            "🚀 发送",
            use_container_width=True,
            help="发送问题给医疗助手"
        )
    with col2:
        st.caption("📝 建议详细描述症状，按 Enter 换行，Ctrl+Enter 发送")

# 处理用户输入
if submit_button and user_input.strip():
    # 添加用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": user_input.strip(),
        "timestamp": datetime.now().isoformat()
    })
    
    # 生成助手回复
    with st.spinner("🤖 医疗助手正在分析中，请稍候..."):
        start_time = time.time()
        response = model.generate_response(
            user_input.strip(),
            max_tokens=800  # 固定参数
        )
        generation_time = time.time() - start_time
    
    # 添加助手消息
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().isoformat(),
        "generation_time": round(generation_time, 2)
    })
    
    # 显示生成时间
    st.sidebar.info(f"⏱️ 生成耗时: {generation_time:.1f}秒")
    
    # 重新渲染页面
    st.rerun()

# 对话统计
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.metric("总对话轮数", len(st.session_state.messages))
with col2:
    user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.metric("用户提问", user_msgs)

# 免责声明
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ 重要医疗免责声明：</strong><br>
    1. 本助手由人工智能驱动，提供的信息仅供参考和教育目的<br>
    2. <strong>不能替代专业医疗建议、诊断或治疗</strong><br>
    3. 如有紧急医疗情况，请立即联系当地急救服务或前往最近医院<br>
    4. 在使用任何药物或治疗方案前，必须咨询专业医生<br>
    5. 模型生成内容可能存在不准确或过时信息，请谨慎参考<br>
    6. 对于因使用本助手提供的信息而导致的任何后果，开发者不承担任何责任
</div>
""", unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.caption(f"""
<div style="text-align: center; color: #757575;">
    医疗智能助手 v1.0 | 基于 Qwen2.5-7B-Instruct 微调 | {datetime.now().year}
</div>
""", unsafe_allow_html=True)

# 运行提示
st.markdown("""
<script>
    // 自动聚焦到输入框
    setTimeout(function() {
        var textarea = document.querySelector('textarea');
        if (textarea) {
            textarea.focus();
        }
    }, 500);
</script>
""", unsafe_allow_html=True)