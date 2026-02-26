"""
SFT训练正确性验证脚本
检查CoT格式、数据处理、损失函数等关键环节
"""

import sys
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk, Dataset
from typing import Dict, List, Tuple

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.processor import games_to_training_format, PROMPT_TEMPLATE_WITH_THINKING


def check_raw_data_format(raw_file: str) -> Dict:
    """检查原始数据格式"""
    print("\n" + "="*70)
    print("1. 检查原始数据格式")
    print("="*70)

    with open(raw_file, 'r', encoding='utf-8') as f:
        game = json.load(f)

    # 检查必需字段
    required_fields = ['game_id', 'difficulty', 'states', 'final_score', 'max_tile', 'total_steps']
    missing_fields = [f for f in required_fields if f not in game]

    if missing_fields:
        print(f"❌ 缺少字段: {missing_fields}")
        return {"status": "error", "missing_fields": missing_fields}

    print(f"✅ Game ID: {game['game_id']}")
    print(f"✅ Difficulty: {game['difficulty']}")
    print(f"✅ States count: {len(game['states'])}")
    print(f"✅ Final score: {game['final_score']}")
    print(f"✅ Max tile: {game['max_tile']}")

    # 检查第一个状态
    first_state = game['states'][0]
    print(f"\n第一个状态:")
    print(f"  - State: {first_state['state'][:50]}...")
    print(f"  - Action: {first_state['action']}")
    print(f"  - Action ID: {first_state['action_id']}")

    # 检查是否有thinking字段
    has_thinking = 'thinking' in first_state
    if has_thinking:
        print(f"  - Thinking: {first_state['thinking'][:50]}...")
        print(f"✅ 数据包含 Chain-of-Thinking 字段")
    else:
        print(f"⚠️ 数据不包含 thinking 字段")

    return {
        "status": "success",
        "has_thinking": has_thinking,
        "num_states": len(game['states'])
    }


def check_data_processing(raw_dir: str, use_thinking: bool = True) -> Dict:
    """检查数据处理流程"""
    print("\n" + "="*70)
    print("2. 检查数据处理流程")
    print("="*70)

    try:
        # 处理数据
        print(f"处理数据 (use_thinking={use_thinking})...")
        dataset = games_to_training_format(
            games_dir=raw_dir,
            use_thinking=use_thinking,
            apply_chat_template=True,
            base_model="Qwen/Qwen3-1.7B"
        )

        print(f"✅ 生成了 {len(dataset)} 个训练样本")

        # 检查第一个样本
        sample = dataset[0]
        print(f"\n第一个样本:")
        print(f"  - Text length: {len(sample['text'])} chars")
        print(f"  - Text preview:\n{sample['text'][:500]}...")

        # 检查是否包含正确的token
        has_think_tag = "<think>" in sample['text']
        has_think_end = "</think>" in sample['text']

        if use_thinking:
            if has_think_tag and has_think_end:
                print(f"✅ 包含正确的 CoT 标签: <think>...</think>")
            else:
                print(f"❌ 缺少 CoT 标签")
                print(f"  - <think>: {has_think_tag}")
                print(f"  - </think>: {has_think_end}")
                return {"status": "error", "missing_cot_tags": True}

        # 检查动作是否在最后
        valid_actions = ['上', '右', '下', '左']
        ends_with_action = any(sample['text'].rstrip().endswith(action) for action in valid_actions)
        if ends_with_action:
            print(f"✅ 样本以动作结尾")
        else:
            print(f"⚠️ 样本未以动作结尾")

        return {
            "status": "success",
            "num_samples": len(dataset),
            "sample_text": sample['text']
        }

    except Exception as e:
        print(f"❌ 数据处理失败: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def check_tokenizer_and_chat_template(text: str, model_name: str = "Qwen/Qwen3-1.7B") -> Dict:
    """检查tokenizer和chat template"""
    print("\n" + "="*70)
    print("3. 检查 Tokenizer 和 Chat Template")
    print("="*70)

    try:
        print(f"加载 tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_ids = tokenizer.convert_ids_to_tokens(tokens)

        print(f"✅ Tokenization 成功")
        print(f"  - 总token数: {len(tokens)}")
        print(f"  - 前50个tokens: {token_ids[:50]}")

        # 检查特殊token
        # <think> token ID 应该是 151652
        # </think> token ID 应该是 151653
        think_token_id = tokenizer.convert_tokens_to_ids("<think>")
        think_end_token_id = tokenizer.convert_tokens_to_ids("</think>")

        print(f"\n特殊Token ID:")
        print(f"  - <think>: {think_token_id}")
        print(f"  - </think>: {think_end_token_id}")

        if think_token_id == 151652:
            print(f"✅ <think> token ID 正确: 151652")
        else:
            print(f"⚠️ <think> token ID 不是 151652")

        if think_end_token_id == 151653:
            print(f"✅ </think> token ID 正确: 151653")
        else:
            print(f"⚠️ </think> token ID 不是 151653")

        # 检查文本中是否包含这些token
        if think_token_id in tokens:
            idx = tokens.index(think_token_id)
            print(f"✅ <think> token 出现在位置 {idx}")
        else:
            print(f"⚠️ <think> token 未在文本中找到")

        if think_end_token_id in tokens:
            idx = tokens.index(think_end_token_id)
            print(f"✅ </think> token 出现在位置 {idx}")
        else:
            print(f"⚠️ </think> token 未在文本中找到")

        return {
            "status": "success",
            "num_tokens": len(tokens),
            "think_token_id": think_token_id,
            "think_end_token_id": think_end_token_id
        }

    except Exception as e:
        print(f"❌ Tokenizer 检查失败: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def check_loss_computation(text: str, model_name: str = "Qwen/Qwen3-1.7B") -> Dict:
    """检查损失计算"""
    print("\n" + "="*70)
    print("4. 检查损失计算")
    print("="*70)

    try:
        print(f"加载模型: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # 使用CPU避免显存问题
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )

        model.eval()

        # Tokenize
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings["input_ids"]

        print(f"Input shape: {input_ids.shape}")

        # 计算loss
        with torch.no_grad():
            outputs = model(
                input_ids,
                labels=input_ids
            )
            loss = outputs.loss

        print(f"✅ 损失计算成功")
        print(f"  - Loss value: {loss.item():.4f}")
        print(f"  - Perplexity: {torch.exp(loss).item():.4f}")

        # 检查CoT部分的损失
        # 找到 <think> 和 </think> 的位置
        think_id = tokenizer.convert_tokens_to_ids("<think>")
        think_end_id = tokenizer.convert_tokens_to_ids("</think>")

        input_ids_list = input_ids[0].tolist()

        if think_id in input_ids_list and think_end_id in input_ids_list:
            think_start = input_ids_list.index(think_id)
            think_end = input_ids_list.index(think_end_id) + 1

            print(f"\nCoT 部分:")
            print(f"  - 起始位置: {think_start}")
            print(f"  - 结束位置: {think_end}")
            print(f"  - Token数: {think_end - think_start}")

            # 只计算CoT部分的损失
            cot_labels = input_ids.clone()
            cot_labels[0, :think_start] = -100
            cot_labels[0, think_end:] = -100

            with torch.no_grad():
                cot_outputs = model(input_ids, labels=cot_labels)
                cot_loss = cot_outputs.loss

            print(f"  - CoT Loss: {cot_loss.item():.4f}")

            # 只计算答案部分的损失
            answer_labels = input_ids.clone()
            answer_labels[0, :think_end] = -100

            with torch.no_grad():
                answer_outputs = model(input_ids, labels=answer_labels)
                answer_loss = answer_outputs.loss

            print(f"\n答案部分:")
            print(f"  - 起始位置: {think_end}")
            print(f"  - Token数: {len(input_ids_list) - think_end}")
            print(f"  - Answer Loss: {answer_loss.item():.4f}")

            return {
                "status": "success",
                "total_loss": loss.item(),
                "cot_loss": cot_loss.item(),
                "answer_loss": answer_loss.item(),
                "cot_token_ratio": (think_end - think_start) / len(input_ids_list)
            }
        else:
            print(f"⚠️ 未找到 CoT token，无法分别计算损失")
            return {
                "status": "success",
                "total_loss": loss.item()
            }

    except Exception as e:
        print(f"❌ 损失计算检查失败: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def check_training_config() -> Dict:
    """检查训练配置"""
    print("\n" + "="*70)
    print("5. 检查训练配置")
    print("="*70)

    try:
        from src.models.trl_train import SFTConfig

        # 默认配置
        config = SFTConfig(
            output_dir="./checkpoints/sft",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            dataset_text_field="text",
            max_seq_length=512,
        )

        print(f"✅ 训练配置:")
        print(f"  - Output dir: {config.output_dir}")
        print(f"  - Epochs: {config.num_train_epochs}")
        print(f"  - Batch size: {config.per_device_train_batch_size}")
        print(f"  - Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"  - Learning rate: {config.learning_rate}")
        print(f"  - Dataset text field: {config.dataset_text_field}")
        print(f"  - Max seq length: {config.max_seq_length}")
        print(f"  - FP16: {config.fp16}")

        # 检查关键配置
        issues = []

        if config.dataset_text_field != "text":
            issues.append("dataset_text_field 应该是 'text'")

        if config.max_seq_length < 512:
            issues.append("max_seq_length 可能太短，无法包含完整的CoT")

        if issues:
            print(f"\n⚠️ 配置问题:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\n✅ 配置看起来正常")

        return {
            "status": "success",
            "config": {
                "epochs": config.num_train_epochs,
                "batch_size": config.per_device_train_batch_size,
                "learning_rate": config.learning_rate,
                "max_seq_length": config.max_seq_length
            }
        }

    except Exception as e:
        print(f"❌ 配置检查失败: {e}")
        return {"status": "error", "error": str(e)}


def main():
    """主函数"""
    print("\n" + "="*70)
    print("SFT训练正确性验证")
    print("="*70)

    results = {}

    # 1. 检查原始数据
    raw_file = "data/raw/game_000000.json"
    if Path(raw_file).exists():
        results["raw_data"] = check_raw_data_format(raw_file)
    else:
        print(f"⚠️ 原始数据文件不存在: {raw_file}")

    # 2. 检查数据处理（带CoT）
    raw_dir = "data/raw"
    if Path(raw_dir).exists():
        results["data_processing_cot"] = check_data_processing(raw_dir, use_thinking=True)

        # 获取处理后的文本
        if results["data_processing_cot"]["status"] == "success":
            sample_text = results["data_processing_cot"]["sample_text"]

            # 3. 检查tokenizer
            results["tokenizer"] = check_tokenizer_and_chat_template(sample_text)

            # 4. 检查损失计算
            print("\n注意: 损失计算需要加载模型，可能需要几分钟...")
            user_input = input("是否继续检查损失计算? (y/n): ")
            if user_input.lower() == 'y':
                results["loss"] = check_loss_computation(sample_text)
            else:
                print("跳过损失计算检查")
    else:
        print(f"⚠️ 原始数据目录不存在: {raw_dir}")

    # 5. 检查训练配置
    results["config"] = check_training_config()

    # 总结
    print("\n" + "="*70)
    print("验证总结")
    print("="*70)

    all_passed = True
    for key, result in results.items():
        if result["status"] == "success":
            print(f"✅ {key}: 通过")
        else:
            print(f"❌ {key}: 失败")
            all_passed = False

    if all_passed:
        print("\n✅ 所有检查通过！SFT训练配置正确。")
    else:
        print("\n❌ 存在问题，请检查上述错误信息。")


if __name__ == "__main__":
    main()
