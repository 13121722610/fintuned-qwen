# 【医疗领域大模型微调项目：基于 Qwen2.5-7B 的医疗问答专家系统】

一个基于 Qwen2.5-7B 大语言模型的医疗领域微调项目，通过监督微调（SFT）和 LoRA 参数高效微调技术，将通用大模型转化为专业的医疗问答助手。  

# 📋 项目概述

本项目旨在解决通用大语言模型在医疗领域的"幻觉"问题和回答泛化问题。通过对 Qwen2.5-7B 模型进行领域特定微调，使其能够生成具有专业结构（病情分析、原因分析、治疗建议）的医疗建议。  

主要特性  

🏥 专业医疗问答：针对医疗领域进行优化，提供专业建议  
🎯 结构化输出：生成"病情分析+原因分析+治疗建议"三段式回答  
⚡ 高效微调：使用 LoRA 技术减少训练参数和资源消耗  
🌐 交互界面：提供 Web 交互界面，方便用户使用  
📊 完整流程：包含数据清洗、优化、训练、评估全流程  

# 📁 项目结构
fintuned-qwen/  
├── data/                          # 数据集  
│   ├── data-10k.json             #  优化后的数据集  
├── medical_data_optimizer/       # 数据集优化代码  
│   ├── optimizer.py              # 数据优化核心代码  
│   └── run_optimization.py       # 数据优化执行脚本  
├── processed_data/               # 预处理后的训练数据  
│   ├── train.json               # 训练集 (90%)  
│   └── validation.json          # 验证集 (10%)  
├── scripts/                      # 脚本文件  
│   ├── prepare_data.py          # 数据准备脚本  
│   ├── finetune_medical.py      # 模型微调脚本  
│   ├── test_finetuned.py        # 微调后模型测试  
│   └── test_qwen2-5.py          # 原始模型测试  
├── finetuned_model_20251212_090808/  # 微调后的模型  
│   ├── adapter_model/           # LoRA 适配器权重  
│   ├── plots/                   # 训练损失曲线等图表  
│   └── ...
├── web_launcher.py              # Web 应用启动器  
├── medical_web_app.py           # Web 应用主程序  
├── requirements.txt             # Python 依赖包  

注：由于优化后的数据集和微调后的模型大小过大，因此没有上传到github。  

# 🚀 快速开始

环境配置  

克隆项目  
git clone https://github.com/13121722610/fintuned-qwen.git  
cd fintuned-qwen  

安装依赖  
pip install -r requirements.txt  

模型训练  
python scripts/finetune_medical.py  

测试微调后模型  
python scripts/test_finetuned.py  

对比原始模型  
python scripts/test_qwen2-5.py  

启动 Web 应用  
python web_launcher.py  
然后在浏览器中访问 http://localhost:7860 使用交互界面。  

# 📊 实验结果

微调前后对比

微调前（原始 Qwen2.5-7B）：  
<img width="924" height="163" alt="2541765603863_ pic" src="https://github.com/user-attachments/assets/f2838b3b-081f-4e98-bd99-7f92ca0ccf04" />

回答较为泛化，缺乏专业性  
无固定回答格式  
可能产生"幻觉"或不准确信息  

微调后（医疗专用模型）：  
<img width="1355" height="785" alt="2491765603090_ pic" src="https://github.com/user-attachments/assets/5dd3df50-e6d0-482d-8946-95c9b57f26e2" />

✅ 回答具有专业医疗知识  
✅ 结构化输出："病情分析 + 原因分析 + 治疗建议"  
✅ 更详细、更准确的医疗建议  
✅ 减少了幻觉现象  

# 📞 联系方式

如有问题或建议，请通过以下方式联系：  
电话/微信：13121722610  
Email: 13121722610@163.com  

免责声明：本项目生成的医疗建议仅供参考，不能替代专业医疗诊断。如有医疗问题，请咨询合格的专业医生。  
