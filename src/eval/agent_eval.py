"""Unified evaluator for rule/random baselines and LLM agents on 2048.

Supports:
- Rule-based baseline agent
- Pure random baseline agent
- LLM agent with thinking on/off
- Quick evaluation mode
- Per-game terminal visualization mode
"""

from __future__ import annotations

import ast
import argparse
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.data_gen.generator import ThinkingHeuristicPlayer
from src.data_gen.prompting import format_inference_prompt
from src.envs.game_2048 import ACTION_MAP, Game2048, parse_action_from_non_think_text

# Hugging Face mirror defaults (honor existing env if user already set it).
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel

    PEFT_AVAILABLE = True
except ImportError:
    PeftModel = None
    PEFT_AVAILABLE = False

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    LLM = None
    SamplingParams = None
    VLLM_AVAILABLE = False

try:
    from vllm.lora.request import LoRARequest

    VLLM_LORA_AVAILABLE = True
except ImportError:
    LoRARequest = None
    VLLM_LORA_AVAILABLE = False


DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

THINKING_SAMPLING_DEFAULTS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
}

NON_THINKING_SAMPLING_DEFAULTS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
}

# Align random-eval opening settings with the environment defaults/rules.
RANDOM_EVAL_INIT_TILES_MIN = 2
RANDOM_EVAL_INIT_TILES_MAX = 3
RANDOM_EVAL_INIT_PROB_4 = 0.1


@dataclass
class AgentDecision:
    action: int
    raw_response: str = ""
    thinking: str = ""


class AgentBase:
    """Common interface for all agents."""

    name: str = "agent"

    def decide(self, game: Game2048, state_text: str, temperature: Optional[float]) -> AgentDecision:
        raise NotImplementedError

    def decide_batch(
        self,
        games: List[Game2048],
        state_texts: List[str],
        temperature: Optional[float],
    ) -> List[AgentDecision]:
        decisions: List[AgentDecision] = []
        for game, state_text in zip(games, state_texts):
            decisions.append(self.decide(game=game, state_text=state_text, temperature=temperature))
        return decisions


class RandomBaselineAgent(AgentBase):
    name = "random_baseline"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def decide(self, game: Game2048, state_text: str, temperature: Optional[float]) -> AgentDecision:
        valid_actions = game.get_valid_actions()
        if not valid_actions:
            return AgentDecision(action=0, raw_response="<no_valid_action>")
        action = self.rng.choice(valid_actions)
        return AgentDecision(action=action, raw_response=ACTION_MAP[action])


class RuleBaselineAgent(AgentBase):
    name = "rule_baseline"

    def __init__(self, difficulty: str = "advanced", seed: Optional[int] = None, with_thinking: bool = True):
        self.player = ThinkingHeuristicPlayer(difficulty=difficulty, seed=seed, enable_diversity=False)
        self.difficulty = difficulty
        self.with_thinking = with_thinking

    def decide(self, game: Game2048, state_text: str, temperature: Optional[float]) -> AgentDecision:
        if self.with_thinking:
            thinking, action = self.player.choose_action_with_thinking(game)
            return AgentDecision(action=action, raw_response=ACTION_MAP[action], thinking=thinking)
        action = self.player.choose_action(game)
        return AgentDecision(action=action, raw_response=ACTION_MAP[action])


class LLMAgent(AgentBase):
    name = "llm"

    def __init__(
        self,
        model_path: Optional[str],
        base_model: str,
        is_base_model: bool,
        use_thinking: bool,
        presence_penalty: float,
        use_vllm: bool,
        vllm_quantization: Optional[str],
        load_in_4bit: bool,
        load_in_8bit: bool,
        device: str,
    ):
        self.model_path = model_path
        self.base_model = base_model
        self.is_base_model = is_base_model
        self.use_thinking = use_thinking
        self.presence_penalty = float(np.clip(float(presence_penalty), 0.0, 2.0))
        self.model_type = "base_model" if is_base_model or model_path is None else "finetuned"

        self.use_vllm = bool(use_vllm and VLLM_AVAILABLE)
        self.vllm_model = None
        self.vllm_lora_request = None

        if use_vllm and not VLLM_AVAILABLE:
            print("[LLM] vLLM 未安装，回退到 transformers 模式")

        # Align with latest evaluator policy: default to BF16/FP16 and avoid bnb quantization.
        if load_in_4bit or load_in_8bit:
            print("[LLM] 已忽略 load_in_4bit/load_in_8bit：评测默认使用 BF16/FP16，不使用 bitsandbytes 量化")

        if self.use_vllm:
            self._init_vllm(vllm_quantization)
        else:
            self._init_standard(device=device)

    @staticmethod
    def _is_lora_adapter_checkpoint(model_path: Optional[str]) -> bool:
        if not model_path:
            return False
        return (Path(model_path) / "adapter_config.json").exists()

    @staticmethod
    def _resolve_base_model_from_adapter(adapter_path: str) -> str:
        adapter_cfg = Path(adapter_path) / "adapter_config.json"
        if not adapter_cfg.exists():
            raise ValueError(f"Not a LoRA adapter checkpoint: {adapter_path}")
        with open(adapter_cfg, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        base = cfg.get("base_model_name_or_path")
        if not base:
            raise ValueError(f"Cannot find base_model_name_or_path in {adapter_cfg}")
        return base

    @staticmethod
    def _resolve_lora_rank_from_adapter(adapter_path: str) -> Optional[int]:
        adapter_cfg = Path(adapter_path) / "adapter_config.json"
        if not adapter_cfg.exists():
            return None
        try:
            with open(adapter_cfg, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            return None
        rank = cfg.get("r")
        if rank is None:
            return None
        try:
            rank_i = int(rank)
        except Exception:
            return None
        return rank_i if rank_i > 0 else None

    @staticmethod
    def _normalize_vllm_quantization(quantization: Optional[str]) -> Optional[str]:
        if not quantization:
            return None
        q = str(quantization).strip().lower()
        alias = {
            "int8": "bitsandbytes",
            "bnb": "bitsandbytes",
        }
        return alias.get(q, q)

    @staticmethod
    def _ensure_vllm_multiproc_spawn() -> None:
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    def _init_vllm(self, quantization: Optional[str]) -> None:
        self._ensure_vllm_multiproc_spawn()
        model_ref = self.base_model
        if self.is_base_model or not self.model_path:
            model_ref = self.base_model
            print(f"[LLM] vLLM 加载基座模型: {model_ref}")
        elif self._is_lora_adapter_checkpoint(self.model_path):
            model_ref = self._resolve_base_model_from_adapter(self.model_path)
            print(f"[LLM] vLLM 加载基座模型: {model_ref}")
            print(f"[LLM] 挂载 LoRA adapter: {self.model_path}")
        else:
            model_ref = self.model_path
            print(f"[LLM] vLLM 加载完整模型: {model_ref}")

        kwargs = {
            "model": model_ref,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 1024,
            "tensor_parallel_size": 1,
            "trust_remote_code": True,
        }

        if self.model_path and (not self.is_base_model) and self._is_lora_adapter_checkpoint(self.model_path):
            if not VLLM_LORA_AVAILABLE:
                raise RuntimeError("vLLM LoRARequest 不可用，请升级 vLLM 或关闭 --use_vllm")
            kwargs["enable_lora"] = True
            lora_rank = self._resolve_lora_rank_from_adapter(self.model_path)
            if lora_rank is not None:
                kwargs["max_lora_rank"] = max(16, lora_rank)
                print(f"[LLM] vLLM max_lora_rank={kwargs['max_lora_rank']} (adapter r={lora_rank})")

        normalized_quant = self._normalize_vllm_quantization(quantization)
        if normalized_quant:
            if normalized_quant != str(quantization).strip().lower():
                print(f"[LLM] vLLM 量化别名映射: {quantization} -> {normalized_quant}")
            kwargs["quantization"] = normalized_quant

        try:
            self.vllm_model = LLM(**kwargs)
        except Exception as exc:
            if normalized_quant and "quantization" in str(exc).lower():
                print(f"[LLM] vLLM quantization={normalized_quant} 初始化失败，回退到无量化 vLLM")
                kwargs.pop("quantization", None)
                self.vllm_model = LLM(**kwargs)
            elif "Cannot re-initialize CUDA in forked subprocess" in str(exc):
                raise RuntimeError(
                    "vLLM 初始化失败（CUDA+fork），请设置 VLLM_WORKER_MULTIPROC_METHOD=spawn 或关闭 --use_vllm"
                ) from exc
            else:
                raise

        self.tokenizer = self.vllm_model.get_tokenizer()

        if self.model_path and (not self.is_base_model) and self._is_lora_adapter_checkpoint(self.model_path):
            self.vllm_lora_request = LoRARequest("agent_eval_lora", 1, self.model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.tokenizer, "padding_side", None) != "left":
            self.tokenizer.padding_side = "left"

    def _init_standard(self, device: str) -> None:
        is_adapter = bool(self.model_path) and self._is_lora_adapter_checkpoint(self.model_path)

        if self.is_base_model or not self.model_path:
            model_to_load = self.base_model
            tokenizer_to_load = self.base_model
            need_lora = False
            print(f"[LLM] transformers 加载基座模型: {model_to_load}")
        elif is_adapter:
            model_to_load = self._resolve_base_model_from_adapter(self.model_path)
            tokenizer_to_load = model_to_load
            need_lora = True
            print(f"[LLM] transformers 加载 LoRA 基座: {model_to_load}")
            print(f"[LLM] LoRA 路径: {self.model_path}")
        else:
            model_to_load = self.model_path
            tokenizer_to_load = self.model_path
            need_lora = False
            print(f"[LLM] transformers 加载完整模型: {model_to_load}")

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if use_bf16 else torch.float16,
            "device_map": device,
            "trust_remote_code": True,
        }

        self.model = AutoModelForCausalLM.from_pretrained(model_to_load, **model_kwargs)
        print(f"[LLM] 推理精度: {'bf16' if use_bf16 else 'fp16'}")

        if need_lora:
            if not PEFT_AVAILABLE:
                raise RuntimeError("需要 peft 才能加载 LoRA checkpoint")
            self.model = PeftModel.from_pretrained(self.model, self.model_path, is_trainable=False)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_to_load, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.tokenizer, "padding_side", None) != "left":
            self.tokenizer.padding_side = "left"

        self.model.eval()

    @staticmethod
    def _extract_thinking(response: str) -> str:
        if not response:
            return ""

        think_match = re.search(r"<think>\s*(.*?)\s*</think>", response, flags=re.S | re.I)
        if think_match:
            return think_match.group(1).strip()

        if "分析：" in response and "着法：" in response:
            try:
                start = response.index("分析：") + len("分析：")
                end = response.index("着法：")
                return response[start:end].strip()
            except ValueError:
                return ""

        return ""

    def _resolve_sampling(self, temperature: Optional[float]) -> Dict[str, float]:
        defaults = THINKING_SAMPLING_DEFAULTS if self.use_thinking else NON_THINKING_SAMPLING_DEFAULTS
        resolved_temperature = defaults["temperature"] if temperature is None else max(float(temperature), 1e-3)
        return {
            "temperature": resolved_temperature,
            "top_p": defaults["top_p"],
            "top_k": defaults["top_k"],
            "min_p": defaults["min_p"],
        }

    def decide(self, game: Game2048, state_text: str, temperature: Optional[float]) -> AgentDecision:
        return self.decide_batch(
            games=[game],
            state_texts=[state_text],
            temperature=temperature,
        )[0]

    def decide_batch(
        self,
        games: List[Game2048],
        state_texts: List[str],
        temperature: Optional[float],
    ) -> List[AgentDecision]:
        if not state_texts:
            return []
        if self.use_vllm:
            actions, responses = self._predict_batch_vllm(state_texts, temperature)
        else:
            actions, responses = self._predict_batch_standard(state_texts, temperature)

        decisions: List[AgentDecision] = []
        for action, response in zip(actions, responses):
            thinking = self._extract_thinking(response) if self.use_thinking else ""
            decisions.append(
                AgentDecision(action=int(action), raw_response=response, thinking=thinking)
            )
        return decisions

    def _predict_batch_vllm(
        self,
        state_texts: List[str],
        temperature: Optional[float],
    ) -> Tuple[List[int], List[str]]:
        sampling = self._resolve_sampling(temperature)
        prompts = [
            format_inference_prompt(
                tokenizer=self.tokenizer,
                state_text=state_text,
                use_thinking=self.use_thinking,
            )
            for state_text in state_texts
        ]
        sampling_params = SamplingParams(
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            top_k=sampling["top_k"],
            min_p=sampling["min_p"],
            max_tokens=256 if self.use_thinking else 64,
            presence_penalty=self.presence_penalty,
        )

        if self.vllm_lora_request is not None:
            outputs = self.vllm_model.generate(
                prompts,
                sampling_params,
                lora_request=self.vllm_lora_request,
                use_tqdm=False,
            )
        else:
            outputs = self.vllm_model.generate(prompts, sampling_params, use_tqdm=False)

        responses: List[str] = []
        actions: List[int] = []
        for out in outputs:
            response = out.outputs[0].text if out.outputs else ""
            responses.append(response)
            parsed = parse_action_from_non_think_text(response)
            actions.append(int(parsed) if parsed is not None else -1)
        return actions, responses

    def _predict_batch_standard(
        self,
        state_texts: List[str],
        temperature: Optional[float],
    ) -> Tuple[List[int], List[str]]:
        sampling = self._resolve_sampling(temperature)
        prompts = [
            format_inference_prompt(
                tokenizer=self.tokenizer,
                state_text=state_text,
                use_thinking=self.use_thinking,
            )
            for state_text in state_texts
        ]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generate_kwargs = {
                **inputs,
                "max_new_tokens": 768 if self.use_thinking else 64,
                "temperature": sampling["temperature"],
                "top_p": sampling["top_p"],
                "top_k": sampling["top_k"],
                "do_sample": True,  # 禁止贪心解码
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            generation_config_dict = (
                self.model.generation_config.to_dict()
                if hasattr(self.model, "generation_config") and hasattr(self.model.generation_config, "to_dict")
                else {}
            )
            if "min_p" in generation_config_dict:
                generate_kwargs["min_p"] = sampling["min_p"]
            outputs = self.model.generate(**generate_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        responses: List[str] = []
        actions: List[int] = []
        for seq in outputs:
            response = self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            responses.append(response)
            parsed = parse_action_from_non_think_text(response)
            actions.append(int(parsed) if parsed is not None else -1)
        return actions, responses


def derive_episode_seed(
    base_seed: Optional[int],
    index: int,
    tag: str = "",
    sample: Optional[Dict] = None,
) -> Optional[int]:
    if base_seed is None:
        return None
    material = f"{int(base_seed)}:{int(index)}:{tag}"
    if sample:
        material += (
            f":{sample.get('source_game_id', '')}"
            f":{sample.get('source_step', '')}"
            f":{sample.get('bucket', '')}"
        )
    digest = hashlib.blake2b(material.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32)


def load_eval_set(path: str) -> List[Dict]:
    samples: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def init_game_from_sample(sample: Dict, seed: Optional[int]) -> Game2048:
    game = Game2048(seed=seed)
    grid = np.array(ast.literal_eval(sample["state"]), dtype=int)
    if grid.shape != (4, 4):
        raise ValueError(f"Invalid state shape: {grid.shape}")
    game.grid = grid
    game.score = int(sample.get("start_score", 0))
    game.game_over = game._is_game_over()
    return game


def setup_global_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def board_to_lines(grid: np.ndarray) -> List[str]:
    lines: List[str] = []
    for row in grid:
        line = " ".join(f"{int(x):4d}" if int(x) > 0 else "   ." for x in row)
        lines.append(line)
    return lines


def print_step_view(
    game_idx: int,
    step_idx: int,
    game: Game2048,
    decision: AgentDecision,
    applied_action: int,
    is_legal: bool,
    valid_actions: List[int],
) -> None:
    print(f"\n[可视化] game={game_idx + 1} step={step_idx + 1}")
    print("-" * 44)
    for line in board_to_lines(game.grid):
        print(line)
    pred_name = ACTION_MAP.get(decision.action, "未知")
    applied_name = ACTION_MAP.get(applied_action, "未知")
    valid_names = [ACTION_MAP[a] for a in valid_actions]
    print(f"预测动作: {pred_name} | 合法: {is_legal} | 实际执行: {applied_name} | 可选: {valid_names}")
    if decision.thinking:
        thinking = decision.thinking.strip().replace("\n", " ")
        if len(thinking) > 140:
            thinking = thinking[:140] + " ..."
        print(f"思考: {thinking}")


def evaluate_agent(
    agent: AgentBase,
    num_games: int,
    max_steps: int,
    temperature: Optional[float],
    seed: Optional[int],
    visualize_game: bool,
    visualize_delay: float,
    batch_size: int = 1,
    eval_samples: Optional[List[Dict]] = None,
    verbose: bool = True,
) -> Dict:
    setup_global_seed(seed)

    total_games = len(eval_samples) if eval_samples is not None else int(num_games)
    batch_size = max(1, int(batch_size))

    scores: List[int] = []
    max_tiles: List[int] = []
    legal_move_ratios: List[float] = []
    steps_list: List[int] = []
    bucket_counter: Dict[str, int] = {}

    completed = 0
    for start in range(0, total_games, batch_size):
        end = min(start + batch_size, total_games)
        records: List[Dict] = []

        for game_idx in range(start, end):
            if eval_samples is not None:
                sample = eval_samples[game_idx]
                game_seed = derive_episode_seed(seed, game_idx, tag=agent.name, sample=sample)
                game = init_game_from_sample(sample=sample, seed=game_seed)
                bucket = sample.get("bucket", "unknown")
                bucket_counter[bucket] = bucket_counter.get(bucket, 0) + 1
            else:
                game_seed = derive_episode_seed(seed, game_idx, tag=agent.name)
                episode_rng = random.Random(game_seed)
                init_tiles = episode_rng.randint(
                    RANDOM_EVAL_INIT_TILES_MIN,
                    RANDOM_EVAL_INIT_TILES_MAX,
                )
                game = Game2048(seed=game_seed)
                game.reset(initial_tiles=init_tiles, prob_4=RANDOM_EVAL_INIT_PROB_4)

            records.append(
                {
                    "game": game,
                    "game_idx": game_idx,
                    "done": bool(game.game_over),
                    "total_moves": 0,
                    "legal_moves": 0,
                }
            )

        for step_idx in range(max_steps):
            active_indices = [i for i, rec in enumerate(records) if not rec["done"]]
            if not active_indices:
                break

            active_games = [records[i]["game"] for i in active_indices]
            states = [game._get_state() for game in active_games]
            decisions = agent.decide_batch(games=active_games, state_texts=states, temperature=temperature)

            for rec_idx, decision in zip(active_indices, decisions):
                rec = records[rec_idx]
                game = rec["game"]

                valid_actions = game.get_valid_actions()
                is_legal = decision.action in valid_actions
                applied_action = decision.action

                if is_legal:
                    rec["legal_moves"] += 1
                    _, _, done, _ = game.step(applied_action)
                else:
                    if valid_actions:
                        applied_action = valid_actions[0]
                        _, _, done, _ = game.step(applied_action)
                    else:
                        done = True

                if visualize_game and rec["game_idx"] == 0:
                    print_step_view(
                        game_idx=rec["game_idx"],
                        step_idx=step_idx,
                        game=game,
                        decision=decision,
                        applied_action=applied_action,
                        is_legal=is_legal,
                        valid_actions=valid_actions,
                    )
                    if visualize_delay > 0:
                        time.sleep(visualize_delay)

                rec["total_moves"] += 1
                rec["done"] = bool(done)

        for rec in records:
            game = rec["game"]
            total_moves = int(rec["total_moves"])
            scores.append(int(game.score))
            max_tiles.append(int(game.get_max_tile()))
            legal_move_ratios.append(float(rec["legal_moves"] / total_moves if total_moves > 0 else 0.0))
            steps_list.append(total_moves)
            completed += 1
            if verbose:
                print(
                    f"进度: {completed}/{total_games} | score={game.score} | max_tile={game.get_max_tile()}"
                )

    results = {
        "num_games": total_games,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "max_score": int(np.max(scores)),
        "min_score": int(np.min(scores)),
        "median_score": float(np.median(scores)),
        "mean_max_tile": float(np.mean(max_tiles)),
        "max_tile_reached": int(np.max(max_tiles)),
        "legal_move_rate": float(np.mean(legal_move_ratios)),
        "mean_steps": float(np.mean(steps_list)),
        "scores": scores,
        "max_tiles": max_tiles,
        "legal_move_ratios": legal_move_ratios,
        "steps": steps_list,
    }
    if eval_samples is not None:
        results["bucket_distribution"] = bucket_counter
    return results


def save_results(results: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {output_path}")


def visualize_results(results: Dict, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(results["scores"], bins=20, color="steelblue", edgecolor="black")
    axes[0, 0].set_title("Score Distribution")
    axes[0, 0].set_xlabel("Score")
    axes[0, 0].set_ylabel("Frequency")

    tile_bins = sorted(set(results["max_tiles"]))
    if len(tile_bins) <= 1:
        tile_bins = [0, 64, 128, 256, 512, 1024, 2048, 4096]
    axes[0, 1].hist(results["max_tiles"], bins=tile_bins, color="coral", edgecolor="black")
    axes[0, 1].set_title("Max Tile Distribution")
    axes[0, 1].set_xlabel("Max Tile")

    axes[1, 0].bar(["Legal Move Rate"], [results["legal_move_rate"]], color="lightgreen", edgecolor="black")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title(f"Legal Move Rate: {results['legal_move_rate']*100:.1f}%")

    axes[1, 1].hist(results["steps"], bins=20, color="plum", edgecolor="black")
    axes[1, 1].set_title("Steps Per Game")
    axes[1, 1].set_xlabel("Steps")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"可视化图已保存: {output_path}")


def print_summary(results: Dict) -> None:
    print("\n" + "=" * 52)
    print("AGENT EVALUATION RESULTS")
    print("=" * 52)
    print(f"num_games       : {results['num_games']}")
    print(f"mean_score      : {results['mean_score']:.2f}")
    print(f"std_score       : {results['std_score']:.2f}")
    print(f"median_score    : {results['median_score']:.2f}")
    print(f"max_score       : {results['max_score']}")
    print(f"mean_max_tile   : {results['mean_max_tile']:.2f}")
    print(f"max_tile_reached: {results['max_tile_reached']}")
    print(f"legal_move_rate : {results['legal_move_rate']*100:.2f}%")
    print(f"mean_steps      : {results['mean_steps']:.2f}")
    print("=" * 52)


def build_agent(args: argparse.Namespace) -> AgentBase:
    if args.agent_type == "random":
        return RandomBaselineAgent(seed=args.seed)

    if args.agent_type == "rule":
        return RuleBaselineAgent(
            difficulty=args.rule_difficulty,
            seed=args.seed,
            with_thinking=args.rule_with_thinking,
        )

    # LLM agent
    if not args.is_base_model and not args.model_path:
        raise ValueError("LLM 模式下需要 --model_path，或使用 --is_base_model")

    return LLMAgent(
        model_path=args.model_path,
        base_model=args.base_model,
        is_base_model=args.is_base_model,
        use_thinking=args.use_thinking,
        presence_penalty=args.presence_penalty,
        use_vllm=args.use_vllm,
        vllm_quantization=args.vllm_quantization,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        device=args.device,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 2048 agents: rule/random baselines and LLM (thinking on/off)."
    )

    parser.add_argument("--agent_type", choices=["rule", "random", "llm"], required=True)

    # Rule baseline options
    parser.add_argument(
        "--rule_difficulty",
        choices=["basic", "intermediate", "advanced", "expert"],
        default="advanced",
        help="Rule baseline difficulty when --agent_type rule",
    )
    parser.add_argument(
        "--rule_with_thinking",
        action="store_true",
        help="When rule baseline is used, also generate/display heuristic thinking text",
    )

    # LLM options
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--is_base_model", action="store_true")

    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument(
        "--vllm_quantization",
        type=str,
        default=None,
        help="vLLM quantization: bitsandbytes/awq/gptq (int8 会映射为 bitsandbytes)",
    )
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--device", type=str, default="auto")

    think_group = parser.add_mutually_exclusive_group()
    think_group.add_argument("--use_thinking", dest="use_thinking", action="store_true")
    think_group.add_argument("--no_thinking", dest="use_thinking", action="store_false")
    parser.set_defaults(use_thinking=True)

    # Eval options
    parser.add_argument("--num_games", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1, help="并行评测 batch 大小")
    parser.add_argument("--eval_set", type=str, default=None, help="固定快照评测集路径（jsonl）")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature override (default: thinking=0.6, non-thinking=0.7)",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        help="Presence penalty for supported frameworks (0~2, default: 0)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="data/eval_agents")

    # Quick + visualization
    parser.add_argument("--quick", action="store_true", help="Fast run: default to num_games=8, max_steps=200")
    parser.add_argument(
        "--visualize_game",
        action="store_true",
        help="Terminal visualization for the first game",
    )
    parser.add_argument("--visualize_delay", type=float, default=0.0)
    parser.add_argument("--no_plot", action="store_true", help="Skip saving eval_results.png")

    parser.add_argument("--dry_run", action="store_true", help="Only validate config without loading/running model")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.agent_type == "llm" and args.is_base_model and args.model_path:
        print("⚠️  --is_base_model 已启用，忽略 --model_path")
        args.model_path = None

    if args.quick:
        if args.eval_set is None and args.num_games == 100:
            args.num_games = 8
        if args.max_steps == 1000:
            args.max_steps = 200

    eval_samples: Optional[List[Dict]] = None
    if args.eval_set:
        eval_samples = load_eval_set(args.eval_set)
        if not eval_samples:
            raise ValueError(f"空 eval_set: {args.eval_set}")
        args.num_games = len(eval_samples)

    if args.visualize_game and args.batch_size > 1:
        print("⚠️  --visualize_game 与批量评测冲突，已自动将 --batch_size 设为 1")
        args.batch_size = 1

    if args.dry_run:
        print("dry_run 配置:")
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))
        return

    agent = build_agent(args)

    print("\n评测配置:")
    print(f"  agent_type: {args.agent_type}")
    if args.agent_type == "rule":
        print(f"  rule_difficulty: {args.rule_difficulty}")
    if args.agent_type == "llm":
        print(f"  use_thinking: {args.use_thinking}")
        print(f"  use_vllm: {args.use_vllm}")
        print(f"  presence_penalty: {args.presence_penalty}")
    print(f"  num_games: {args.num_games}")
    print(f"  max_steps: {args.max_steps}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  eval_set: {args.eval_set}")
    if args.temperature is None:
        default_temp = THINKING_SAMPLING_DEFAULTS["temperature"] if args.use_thinking else NON_THINKING_SAMPLING_DEFAULTS["temperature"]
        print(f"  temperature: auto ({default_temp})")
    else:
        print(f"  temperature: {args.temperature}")
    print(f"  seed: {args.seed}")

    results = evaluate_agent(
        agent=agent,
        num_games=args.num_games,
        max_steps=args.max_steps,
        temperature=args.temperature,
        seed=args.seed,
        visualize_game=args.visualize_game,
        visualize_delay=max(0.0, float(args.visualize_delay)),
        batch_size=args.batch_size,
        eval_samples=eval_samples,
        verbose=True,
    )

    results["agent_type"] = args.agent_type
    results["seed"] = args.seed
    results["batch_size"] = int(args.batch_size)
    if args.eval_set:
        results["eval_set_path"] = args.eval_set
    results["temperature"] = (
        args.temperature
        if args.temperature is not None
        else (THINKING_SAMPLING_DEFAULTS["temperature"] if args.use_thinking else NON_THINKING_SAMPLING_DEFAULTS["temperature"])
    )
    results["max_steps"] = args.max_steps

    if args.agent_type == "rule":
        results["rule_difficulty"] = args.rule_difficulty
        results["rule_with_thinking"] = bool(args.rule_with_thinking)
    elif args.agent_type == "llm":
        results["model_path"] = args.model_path
        results["base_model"] = args.base_model
        results["is_base_model"] = bool(args.is_base_model)
        results["use_thinking"] = bool(args.use_thinking)
        results["presence_penalty"] = float(np.clip(float(args.presence_penalty), 0.0, 2.0))
        results["model_type"] = getattr(agent, "model_type", "unknown")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "eval_results.json"
    save_results(results, json_path)

    if not args.no_plot:
        png_path = output_dir / "eval_results.png"
        visualize_results(results, png_path)

    print_summary(results)


if __name__ == "__main__":
    main()
