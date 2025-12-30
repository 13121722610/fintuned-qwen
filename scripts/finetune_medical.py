import torch
import os

# ========== 修复OpenBLAS警告 ==========
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
# =====================================

# 指定使用GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 设置镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("✅ 使用HF镜像: https://hf-mirror.com")

class TrainingPlotter:
    """训练过程图表生成器"""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.train_losses = []          # 训练loss
        self.eval_losses = []           # 验证loss
        self.learning_rates = []        # 学习率
        self.train_steps = []           # 训练步数
        self.eval_steps = []            # 验证步数
        self.logs_history = []          # 完整日志历史
        self.epoch_logs = []            # epoch级别日志
        
    def add_log(self, log_dict, step=None, epoch=None):
        """添加日志"""
        log_dict_copy = log_dict.copy()
        
        # 添加时间戳
        log_dict_copy['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if step is not None:
            log_dict_copy['step'] = step
        if epoch is not None:
            log_dict_copy['epoch'] = epoch
        
        self.logs_history.append(log_dict_copy)
        
        # 提取训练loss
        if 'loss' in log_dict and log_dict['loss'] is not None:
            current_step = len(self.train_losses) + 1
            self.train_losses.append(log_dict['loss'])
            self.train_steps.append(current_step)
            
            # 每50步打印一次训练进度
            if current_step % 50 == 0:
                print(f"📊 训练进度: Step {current_step}, Loss: {log_dict['loss']:.4f}")
        
        # 提取验证loss
        if 'eval_loss' in log_dict and log_dict['eval_loss'] is not None:
            current_eval_step = len(self.eval_losses) + 1
            self.eval_losses.append(log_dict['eval_loss'])
            self.eval_steps.append(current_eval_step)
            
            print(f"📈 Epoch {log_dict.get('epoch', '?')} 验证Loss: {log_dict['eval_loss']:.4f}")
        
        # 提取学习率
        if 'learning_rate' in log_dict:
            self.learning_rates.append(log_dict['learning_rate'])
        
        # 提取epoch信息
        if 'epoch' in log_dict:
            epoch_info = {
                'epoch': log_dict['epoch'],
                'timestamp': log_dict_copy['timestamp']
            }
            if 'loss' in log_dict:
                epoch_info['train_loss'] = log_dict['loss']
            if 'eval_loss' in log_dict:
                epoch_info['eval_loss'] = log_dict['eval_loss']
            self.epoch_logs.append(epoch_info)
    
    def print_epoch_summary(self, epoch, train_loss, eval_loss=None):
        """打印epoch总结"""
        print("\n" + "="*60)
        print(f"🎉 Epoch {epoch} 完成!")
        print(f"   训练步数: {len(self.train_losses)}")
        print(f"   平均训练Loss: {train_loss:.4f}")
        if eval_loss is not None:
            print(f"   验证Loss: {eval_loss:.4f}")
        print("="*60)
    
    def save_all_plots(self):
        """保存所有图表"""
        if not self.train_losses:
            print("⚠️ 没有训练日志数据")
            return
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 训练损失
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(self.train_steps, self.train_losses, 'b-', linewidth=1, alpha=0.7, label='Training Loss')
        
        # 添加平滑曲线
        if len(self.train_losses) > 10:
            window = min(50, len(self.train_losses) // 10)
            smooth_loss = pd.Series(self.train_losses).rolling(window=window, min_periods=1).mean()
            ax1.plot(self.train_steps, smooth_loss, 'r-', linewidth=2, alpha=0.9, label=f'Smoothed (window={window})')
        
        ax1.set_title('Training Loss Progression', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Training Steps', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # 2. 验证损失
        if self.eval_losses:
            ax2 = plt.subplot(2, 2, 2)
            ax2.plot(range(1, len(self.eval_losses)+1), self.eval_losses, 
                    'r-o', linewidth=2, markersize=8, label='Validation Loss')
            ax2.set_title('Validation Loss per Epoch', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Validation Loss', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
            
            # 标记最佳epoch
            if self.eval_losses:
                best_epoch = np.argmin(self.eval_losses) + 1
                best_loss = min(self.eval_losses)
                ax2.plot(best_epoch, best_loss, 'g*', markersize=15, label=f'Best (Epoch {best_epoch})')
                ax2.legend(fontsize=10)
        
        # 3. 学习率变化
        if self.learning_rates:
            ax3 = plt.subplot(2, 2, 3)
            ax3.plot(range(len(self.learning_rates)), self.learning_rates, 'g-', linewidth=2, alpha=0.8)
            ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Logging Steps', fontsize=12)
            ax3.set_ylabel('Learning Rate', fontsize=12)
            ax3.grid(True, alpha=0.3)
        
        # 4. 训练过程摘要
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        
        # 修复格式字符串问题：使用显式的字符串格式化
        initial_loss = self.train_losses[0] if self.train_losses else 0
        final_loss = self.train_losses[-1] if self.train_losses else 0
        loss_decrease = initial_loss - final_loss if len(self.train_losses) > 1 else 0
        
        summary_text = f"""训练过程摘要:

总训练步数: {len(self.train_losses)}
总验证次数: {len(self.eval_losses)}

初始训练Loss: {initial_loss:.4f}
最终训练Loss: {final_loss:.4f}
Loss下降: {loss_decrease:.4f}

训练开始: {self.logs_history[0]['timestamp'] if self.logs_history else 'N/A'}
训练结束: {self.logs_history[-1]['timestamp'] if self.logs_history else 'N/A'}
"""
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(self.output_dir, 'training_plots.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 训练分析图表已保存: {plot_path}")
        
        # 保存日志数据为CSV
        if self.logs_history:
            df = pd.DataFrame(self.logs_history)
            csv_path = os.path.join(self.output_dir, 'training_logs.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"📊 日志数据已保存: {csv_path}")

class EnhancedLoggingCallback(TrainerCallback):
    """增强的日志记录回调"""
    def __init__(self, plotter):
        super().__init__()
        self.plotter = plotter
        self.current_step = 0
        self.current_epoch = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # 记录当前步数
            if state and hasattr(state, 'global_step'):
                self.current_step = state.global_step
                self.current_epoch = state.epoch
            
            # 添加步数信息
            logs['global_step'] = self.current_step
            logs['epoch'] = self.current_epoch
            
            # 传递给plotter
            self.plotter.add_log(logs, step=self.current_step, epoch=self.current_epoch)
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """每个epoch开始时调用"""
        if state:
            self.current_epoch = state.epoch
            print(f"\n{'='*70}")
            print(f"📅 开始 Epoch {int(self.current_epoch)+1}/{args.num_train_epochs}")
            print(f"{'='*70}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """每个epoch结束时调用"""
        if state:
            print(f"\n{'='*70}")
            print(f"🎉 Epoch {int(state.epoch)+1} 完成!")
            print(f"   累计训练步数: {state.global_step}")
            print(f"   学习率: {state.log_history[-1].get('learning_rate', 'N/A') if state.log_history else 'N/A'}")
            print(f"{'='*70}")

def finetune_qwen():
    """微调Qwen2.5-7B-Instruct模型 - 优化加速版"""
    
    print("=" * 70)
    print("🚀 Qwen2.5-7B-Instruct 医疗领域微调（优化加速版）")
    print("=" * 70)
    
    # 1. 设置路径
    base_model = "Qwen/Qwen2.5-7B-Instruct"
    dataset_path = "/amax/home/yhji/LM-Course/processed_data"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/amax/home/yhji/LM-Course/finetuned_model_{timestamp}"
    plots_dir = os.path.join(output_dir, "plots")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 图表目录: {plots_dir}")
    
    # 2. 检查硬件配置
    print("\n🖥️ 硬件配置检查:")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"PyTorch版本: {torch.__version__}")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU数量: {gpu_count}")
        for i in range(gpu_count):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {gpu_props.name}")
            print(f"    显存: {gpu_props.total_memory / 1024**3:.1f} GB")
        print(f"支持bfloat16: {torch.cuda.is_bf16_supported()}")
    else:
        print("⚠️ 警告: 未检测到GPU，使用CPU模式")
    
    # 3. 初始化图表记录器
    plotter = TrainingPlotter(plots_dir)
    
    # 4. 加载数据
    print("\n📂 加载数据集...")
    dataset = load_from_disk(dataset_path)
    
    print(f"✅ 数据加载完成")
    print(f"  训练集: {len(dataset['train'])} 条")
    print(f"  验证集: {len(dataset['test'])} 条")
    
    # 5. 加载tokenizer
    print("\n🔤 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 6. 数据预处理函数
    def tokenize_function(example):
        """Tokenize对话数据"""
        conversations = example["conversations"]
        
        # 使用Qwen的聊天模板
        text = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize - 减少最大长度以加速
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=512,  # 减少到512以加速
            padding=False
        )
        
        # 设置labels（用于计算损失）
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    print("🔧 Tokenizing数据集...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
        num_proc=4  # 多进程处理
    )
    
    # 优化数据集格式
    tokenized_dataset = tokenized_dataset.with_format(
        "torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    
    # 7. 加载模型（关键优化：启用Flash Attention 2）
    print("\n🤖 加载模型...")
    
    # 根据GPU情况选择精度
    if torch.cuda.is_available():
        print(f"🎮 GPU可用: {torch.cuda.device_count()}个")
        
        # 优先使用bfloat16
        if torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            print("📐 使用精度: bfloat16")
        else:
            torch_dtype = torch.float16
            print("📐 使用精度: float16")
        
        # 检查是否支持Flash Attention 2
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                use_cache=False,
                attn_implementation="flash_attention_2",  # 关键加速！
            )
            print("✅ 已启用Flash Attention 2 (大幅加速)")
        except:
            print("⚠️ Flash Attention 2不可用，使用标准注意力")
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                use_cache=False,
            )
    else:
        torch_dtype = torch.float32
        print("⚠️ 使用CPU模式")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map=None,
            trust_remote_code=True,
            use_cache=False,
        )
    
    # 启用输入梯度
    model.enable_input_require_grads()
    
    # 确保所有参数需要梯度
    for param in model.parameters():
        param.requires_grad = True
    
    print("✅ 已启用模型梯度")
    
    # 8. 配置LoRA
    print("\n🎯 配置LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        inference_mode=False
    )
    
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"📊 总参数: {total_params:,}")
    
    # 9. 训练参数 - 优化版
    print("\n⚙️ 设置训练参数（优化加速模式）...")
    
    # 计算总步数
    train_dataset_size = len(tokenized_dataset["train"])
    per_device_batch = 4  # 根据显存调整：2, 4, 8
    gradient_accumulation = 2  # 减少梯度累积步数
    effective_batch = per_device_batch * gradient_accumulation
    
    # 如果显存不足，减少batch size
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 20:  # 小于20GB显存
            per_device_batch = 2
            gradient_accumulation = 4
            effective_batch = per_device_batch * gradient_accumulation
            print(f"⚠️ 显存较小({gpu_memory:.1f}GB)，使用batch_size={per_device_batch}")
    
    total_steps = (train_dataset_size * 3) // effective_batch  # 3个epoch
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=8,  # 评估用更大batch
        gradient_accumulation_steps=gradient_accumulation,
        warmup_steps=100,
        learning_rate=1e-4,
        fp16=torch_dtype == torch.float16,
        bf16=torch_dtype == torch.bfloat16,
        logging_steps=20,               # 减少日志频率
        eval_strategy="steps",          # 按步评估而不是按epoch
        eval_steps=500,                 # 每500步评估一次
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # 新版本用法
        optim="adamw_torch_fused",      # 使用融合优化器加速
        logging_dir=os.path.join(output_dir, "logs"),
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to=[],
        ddp_find_unused_parameters=False,
        logging_first_step=True,
        logging_nan_inf_filter=False,
        
        # 数据加载优化
        dataloader_num_workers=4,       # 增加数据加载进程
        dataloader_prefetch_factor=2,   # 预取数据
        remove_unused_columns=True,     # 移除未用列
        
        # 防过拟合参数
        weight_decay=0.01,
        max_grad_norm=1.0,
        label_smoothing_factor=0.0,
        
        # 内存优化
        eval_accumulation_steps=1,
        # fsdp="auto_wrap" if torch.cuda.device_count() > 1 else None,  # 多GPU时启用
    )
    
    # 10. 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,  # 对齐到8的倍数，加速计算
    )
    
    # 11. 显示训练配置
    print("\n" + "="*70)
    print("📋 训练配置摘要:")
    print("="*70)
    print(f"1. 训练模式: 优化加速模式")
    print(f"   • 训练轮数: {training_args.num_train_epochs}")
    print(f"   • Batch大小: {per_device_batch} × {gradient_accumulation} = {effective_batch}")
    print(f"   • 总训练步数: ~{total_steps}")
    print(f"   • 序列长度: 512")
    
    print(f"\n2. 模型配置:")
    print(f"   • LoRA秩: {lora_config.r}")
    print(f"   • 可训练参数: {trainable_params/total_params*100:.2f}%")
    
    print(f"\n3. 优化配置:")
    print(f"   • Flash Attention 2: 已启用")
    print(f"   • 优化器: {training_args.optim}")
    print(f"   • 数据加载进程: {training_args.dataloader_num_workers}")
    
    print(f"\n4. 策略配置:")
    print(f"   • 评估频率: 每{training_args.eval_steps}步")
    print(f"   • 保存频率: 每{training_args.save_steps}步")
    print(f"   • 日志频率: 每{training_args.logging_steps}步")
    print("="*70)
    
    # 12. 创建Trainer
    print("\n🤖 创建Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        callbacks=[
            EnhancedLoggingCallback(plotter),
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001
            )
        ],
    )
    
    # 单独设置tokenizer
    trainer.tokenizer = tokenizer
    
    # 13. 性能测试（可选）
    if torch.cuda.is_available():
        print("\n⚡ 性能测试...")
        try:
            # 测试一个batch的前向传播
            test_batch = next(iter(trainer.get_train_dataloader()))
            for k, v in test_batch.items():
                if isinstance(v, torch.Tensor):
                    test_batch[k] = v.to(model.device)
            
            import time
            start = time.time()
            with torch.no_grad():
                outputs = model(**test_batch)
            end = time.time()
            print(f"单batch前向时间: {(end-start)*1000:.1f}ms")
        except:
            pass
    
    # 14. 开始训练！
    print("\n" + "="*70)
    print("🔥 开始训练...")
    print("="*70)
    
    try:
        train_result = trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断，保存当前进度...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        plotter.save_all_plots()
        return output_dir
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        return output_dir
    
    # 15. 保存最终模型
    print("\n💾 保存最终模型...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # 保存训练结果
    with open(os.path.join(output_dir, "training_results.json"), "w", encoding='utf-8') as f:
        json.dump(train_result.metrics, f, indent=2, ensure_ascii=False)
    
    # 16. 保存图表
    print("\n📈 生成训练图表...")
    plotter.save_all_plots()
    
    # 17. 最终评估
    print("\n📊 最终评估...")
    eval_results = trainer.evaluate()
    
    # 保存评估结果
    eval_file = os.path.join(output_dir, "final_evaluation.json")
    with open(eval_file, "w", encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("✅ 训练完成！")
    print("="*70)
    print(f"📁 模型目录: {output_dir}")
    print(f"📊 最终验证损失: {eval_results.get('eval_loss', 'N/A'):.4f}")
    print(f"📊 总训练步数: {train_result.global_step}")
    print(f"📊 总训练轮数: {train_result.epoch}")
    print(f"📊 训练时间: {train_result.metrics.get('train_runtime', 0):.1f}秒")
    
    if torch.cuda.is_available():
        print(f"\n🎮 GPU使用统计:")
        for i in range(torch.cuda.device_count()):
            mem_used = torch.cuda.memory_allocated(i) / 1024**3
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {mem_used:.1f}/{mem_total:.1f} GB ({mem_used/mem_total*100:.1f}%)")
    
    # 计算平均速度
    if train_result.metrics.get('train_runtime', 0) > 0:
        steps_per_second = train_result.global_step / train_result.metrics['train_runtime']
        print(f"⚡ 平均速度: {steps_per_second:.2f} 步/秒, {1/steps_per_second:.2f} 秒/步")
    
    return output_dir

if __name__ == "__main__":
    finetune_qwen()
    