#!/usr/bin/env python3
"""
2048游戏可视化评估脚本

功能：
- 窗口化显示2048游戏
- 实时与模型交互
- 显示模型思考过程（thinking模式）
- 支持调整游戏速度

使用：
python scripts/visualize_2048.py --model_path ./checkpoints/sft
"""

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None
    PYGAME_AVAILABLE = False
import torch
import sys
import argparse
import time
from pathlib import Path
from typing import Tuple, Optional, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.envs.game_2048 import Game2048, parse_action_from_text
from src.data.generator import ThinkingHeuristicPlayer

# 尝试导入优化库
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


# 颜色定义
COLORS = {
    'background': (187, 173, 160),
    'grid': (205, 193, 180),
    'empty': (205, 193, 180),
    'text_dark': (119, 110, 101),
    'text_light': (249, 246, 242),
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}

TILE_COLORS = {
    2: {'bg': (238, 228, 218), 'text': (119, 110, 101)},
    4: {'bg': (237, 224, 200), 'text': (119, 110, 101)},
    8: {'bg': (242, 177, 121), 'text': (249, 246, 242)},
    16: {'bg': (245, 149, 99), 'text': (249, 246, 242)},
    32: {'bg': (246, 124, 95), 'text': (249, 246, 242)},
    64: {'bg': (246, 94, 59), 'text': (249, 246, 242)},
    128: {'bg': (237, 207, 114), 'text': (249, 246, 242)},
    256: {'bg': (237, 204, 97), 'text': (249, 246, 242)},
    512: {'bg': (237, 200, 80), 'text': (249, 246, 242)},
    1024: {'bg': (237, 197, 63), 'text': (249, 246, 242)},
    2048: {'bg': (237, 194, 46), 'text': (249, 246, 242)},
}

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


class Game2048Visualizer:
    """2048游戏可视化器"""

    def __init__(
        self,
        model_path: str,
        base_model: str = "Qwen/Qwen3-1.7B",
        cell_size: int = 100,
        delay: float = 0.5,
        use_vllm: bool = True,
        vllm_quant: Optional[str] = "int8",
        agent_type: str = "llm",
        rule_difficulty: str = "advanced",
        seed: Optional[int] = None,
        use_thinking: bool = True,
        temperature: Optional[float] = None,
        presence_penalty: float = 0.0,
    ):
        """
        初始化可视化器

        Args:
            model_path: 模型路径
            base_model: 基座模型名称
            cell_size: 格子大小
            delay: 每步延迟（秒）
            use_vllm: 是否使用vLLM加速
            vllm_quant: vLLM量化方式
        """
        self.cell_size = cell_size
        self.padding = 10
        self.delay = delay

        # 计算窗口大小
        self.grid_size = cell_size * 4 + self.padding * 5
        self.info_width = max(420, int(cell_size * 4.2))
        self.window_width = self.grid_size + self.info_width + 40
        self.window_height = max(self.grid_size + 40, 720)
        self.info_panel_x = self.grid_size + 20
        self.info_panel_y = 10
        self.info_panel_padding = 16
        self.info_panel_inner_width = self.info_width - self.info_panel_padding * 2

        # 初始化pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("2048 AI 可视化")
        self.clock = pygame.time.Clock()

        # 字体
        self._init_fonts()

        # 加载模型
        self.agent_type = agent_type
        self.rule_player = None
        self.use_vllm = False
        self.use_thinking = bool(use_thinking)
        self.temperature = temperature
        self.presence_penalty = float(max(0.0, min(2.0, float(presence_penalty))))

        if self.agent_type in ("rule", "random"):
            difficulty = "random" if self.agent_type == "random" else rule_difficulty
            self.rule_player = ThinkingHeuristicPlayer(
                difficulty=difficulty,
                seed=seed,
                enable_diversity=False,
            )
            print(f"使用基线智能体: {self.agent_type} (difficulty={difficulty})")
        else:
            print(f"加载模型: {model_path}")
            self.use_vllm = use_vllm and VLLM_AVAILABLE
            if self.use_vllm:
                self._load_model_vllm(model_path, vllm_quant)
            else:
                if use_vllm and not VLLM_AVAILABLE:
                    print("⚠️  vLLM未安装，回退到标准模式（安装vLLM可获得更快推理速度）")
                self._load_model_standard(model_path, base_model)

        # 游戏实例
        self.game = Game2048()
        self.game.reset()

        # 统计信息
        self.step_count = 0
        self.thinking_history = []
        self.action_history = []

    def _resolve_sampling(self) -> dict:
        defaults = THINKING_SAMPLING_DEFAULTS if self.use_thinking else NON_THINKING_SAMPLING_DEFAULTS
        resolved_temperature = defaults["temperature"] if self.temperature is None else max(float(self.temperature), 1e-3)
        return {
            "temperature": resolved_temperature,
            "top_p": defaults["top_p"],
            "top_k": defaults["top_k"],
            "min_p": defaults["min_p"],
        }

    def _init_fonts(self):
        """初始化字体，优先选择可显示中文的字体。"""
        cjk_font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ]
        cjk_font_names = [
            "PingFang SC",
            "Hiragino Sans GB",
            "Heiti SC",
            "Songti SC",
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "WenQuanYi Zen Hei",
            "Arial Unicode MS",
            "Arial",
        ]
        latin_font_names = [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
        ]

        self.title_font = self._create_font(38, bold=True, preferred_paths=cjk_font_paths, preferred_names=cjk_font_names)
        self.score_font = self._create_font(24, bold=True, preferred_paths=cjk_font_paths, preferred_names=cjk_font_names)
        self.tile_font = self._create_font(36, bold=True, preferred_paths=[], preferred_names=latin_font_names)
        self.tile_font_small = self._create_font(28, bold=True, preferred_paths=[], preferred_names=latin_font_names)
        self.info_font = self._create_font(18, bold=False, preferred_paths=cjk_font_paths, preferred_names=cjk_font_names)
        self.thinking_font = self._create_font(16, bold=False, preferred_paths=cjk_font_paths, preferred_names=cjk_font_names)

    def _create_font(
        self,
        size: int,
        bold: bool,
        preferred_paths: List[str],
        preferred_names: List[str],
    ):
        for font_path in preferred_paths:
            if Path(font_path).exists():
                try:
                    return pygame.font.Font(font_path, size)
                except Exception:
                    pass

        for name in preferred_names:
            path = pygame.font.match_font(name, bold=bold)
            if path:
                try:
                    return pygame.font.Font(path, size)
                except Exception:
                    pass

        return pygame.font.SysFont(None, size, bold=bold)

    def _wrap_text(self, text: str, font, max_width: int) -> List[str]:
        """按像素宽度换行，适配中文无空格场景。"""
        lines: List[str] = []
        for paragraph in str(text).splitlines():
            if not paragraph:
                lines.append("")
                continue

            current = ""
            for ch in paragraph:
                trial = current + ch
                if font.size(trial)[0] <= max_width:
                    current = trial
                else:
                    if current:
                        lines.append(current)
                    current = ch
            if current:
                lines.append(current)

        if not lines:
            lines = [""]
        return lines

    def _draw_wrapped_text(
        self,
        text: str,
        font,
        color: tuple[int, int, int],
        x: int,
        y: int,
        max_width: int,
        max_lines: Optional[int] = None,
    ) -> int:
        lines = self._wrap_text(text, font, max_width)
        if max_lines is not None and len(lines) > max_lines:
            lines = lines[:max_lines]
            if lines:
                trimmed = lines[-1]
                while trimmed and font.size(trimmed + "...")[0] > max_width:
                    trimmed = trimmed[:-1]
                lines[-1] = (trimmed + "...") if trimmed else "..."

        line_h = font.get_linesize()
        for line in lines:
            surf = font.render(line, True, color)
            self.screen.blit(surf, (x, y))
            y += line_h
        return y

    def _load_model_vllm(self, model_path: str, quantization: Optional[str]):
        """使用vLLM加载模型"""
        print("使用 vLLM 加速推理")

        llm_kwargs = {
            "model": model_path,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 2048,
            "tensor_parallel_size": 1,
            "trust_remote_code": True,
        }

        if quantization:
            llm_kwargs["quantization"] = quantization

        self.vllm_model = LLM(**llm_kwargs)
        self.tokenizer = self.vllm_model.get_tokenizer()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("模型加载完成")

    def _load_model_standard(self, model_path: str, base_model: str):
        """标准方式加载模型"""
        from peft import PeftModel

        # 加载基座模型
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # 加载LoRA权重
        if Path(model_path).exists():
            self.model = PeftModel.from_pretrained(
                self.model,
                model_path,
                is_trainable=False
            )
            print(f"LoRA权重已加载: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        print("模型加载完成")

    def get_model_action(self, state: str) -> Tuple[int, str]:
        """
        使用模型预测动作

        Returns:
            (action_id, thinking_text)
        """
        if self.agent_type in ("rule", "random"):
            if self.rule_player is None:
                return 0, ""
            if self.agent_type == "rule":
                thinking, action = self.rule_player.choose_action_with_thinking(self.game)
                return action, thinking
            action = self.rule_player.choose_action(self.game)
            return action, ""

        content = f"当前2048棋盘状态(4x4数组，0表示空位):\n{state}\n\n请选择最佳着法:"
        messages = [{"role": "user", "content": content}]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.use_thinking
        )

        if self.use_vllm:
            return self._predict_vllm(prompt)
        else:
            return self._predict_standard(prompt)

    def _predict_vllm(self, prompt: str) -> Tuple[int, str]:
        """使用vLLM预测"""
        sampling = self._resolve_sampling()
        sampling_params = SamplingParams(
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            top_k=sampling["top_k"],
            min_p=sampling["min_p"],
            max_tokens=256 if self.use_thinking else 64,
            presence_penalty=self.presence_penalty,
        )

        outputs = self.vllm_model.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text

        # 解析thinking和action
        thinking, content = self._parse_thinking_response(response)
        action = parse_action_from_text(content)

        return action, thinking

    def _predict_standard(self, prompt: str) -> Tuple[int, str]:
        """使用标准方式预测"""
        sampling = self._resolve_sampling()
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generate_kwargs = {
                **inputs,
                "max_new_tokens": 256 if self.use_thinking else 64,
                "temperature": sampling["temperature"],
                "top_p": sampling["top_p"],
                "top_k": sampling["top_k"],
                "do_sample": True,
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

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # 解析thinking和action
        thinking, content = self._parse_thinking_response(response)
        action = parse_action_from_text(content)

        return action, thinking

    def _parse_thinking_response(self, response: str) -> Tuple[str, str]:
        """解析thinking响应"""
        response = response.strip()
        if not self.use_thinking:
            return "", response

        if "分析：" in response and "着法：" in response:
            try:
                analysis_start = response.index("分析：") + len("分析：")
                action_start = response.index("着法：")
                thinking = response[analysis_start:action_start].strip()
                content = response[action_start + len("着法："):].strip()
                return thinking, content
            except ValueError:
                pass

        return "", response

    def draw_grid(self):
        """绘制游戏网格"""
        # 背景
        pygame.draw.rect(
            self.screen,
            COLORS['background'],
            (10, 10, self.grid_size, self.grid_size),
            border_radius=5
        )

        # 绘制格子
        for i in range(4):
            for j in range(4):
                value = self.game.grid[i, j]
                x = 10 + self.padding + j * (self.cell_size + self.padding)
                y = 10 + self.padding + i * (self.cell_size + self.padding)

                # 格子背景
                if value > 0:
                    color = TILE_COLORS.get(value, TILE_COLORS[2048])['bg']
                else:
                    color = COLORS['empty']

                pygame.draw.rect(
                    self.screen,
                    color,
                    (x, y, self.cell_size, self.cell_size),
                    border_radius=5
                )

                # 格子数字
                if value > 0:
                    text_color = TILE_COLORS.get(value, TILE_COLORS[2048])['text']

                    # 根据数字长度选择字体
                    if value < 100:
                        font = self.tile_font
                    elif value < 1000:
                        font = self.tile_font_small
                    else:
                        font = self.tile_font

                    text = font.render(str(value), True, text_color)
                    text_rect = text.get_rect(center=(x + self.cell_size // 2, y + self.cell_size // 2))
                    self.screen.blit(text, text_rect)

    def draw_info(self):
        """绘制信息面板"""
        panel_rect = pygame.Rect(
            self.info_panel_x,
            self.info_panel_y,
            self.info_width,
            self.window_height - 20,
        )
        pygame.draw.rect(self.screen, (242, 237, 229), panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, (210, 198, 182), panel_rect, width=2, border_radius=8)

        x = panel_rect.x + self.info_panel_padding
        y = panel_rect.y + self.info_panel_padding
        max_w = self.info_panel_inner_width

        controls = [
            "控制",
            "SPACE: 暂停/继续",
            "R: 重新开始",
            "ESC: 退出",
        ]
        control_line_h = self.info_font.get_linesize()
        control_block_h = len(controls) * control_line_h + 10
        control_top = panel_rect.bottom - self.info_panel_padding - control_block_h

        agent_map = {
            "llm": "LLM",
            "rule": "规则智能体",
            "random": "随机智能体",
        }
        title_text = f"2048 {agent_map.get(self.agent_type, '智能体')}"
        title_surf = self.title_font.render(title_text, True, COLORS["text_dark"])
        self.screen.blit(title_surf, (x, y))
        y += self.title_font.get_linesize() + 8

        stats = [
            f"分数: {self.game.score}",
            f"最大方块: {self.game.get_max_tile()}",
            f"步数: {self.step_count}",
            f"空位: {self.game.get_empty_cells()}",
        ]
        for item in stats:
            surf = self.score_font.render(item, True, COLORS["text_dark"])
            self.screen.blit(surf, (x, y))
            y += self.score_font.get_linesize() + 6

        y += 6
        action_title = self.info_font.render("最近动作", True, COLORS["text_dark"])
        self.screen.blit(action_title, (x, y))
        y += self.info_font.get_linesize() + 4

        recent_actions = self.action_history[-8:] if self.action_history else []
        action_text = "、".join(recent_actions) if recent_actions else "暂无"
        y = self._draw_wrapped_text(
            text=action_text,
            font=self.info_font,
            color=COLORS["text_dark"],
            x=x,
            y=y,
            max_width=max_w,
            max_lines=3,
        )
        y += 10

        thinking_label = "模型思考" if self.agent_type == "llm" else "策略说明"
        think_title = self.info_font.render(thinking_label, True, COLORS["text_dark"])
        self.screen.blit(think_title, (x, y))
        y += self.info_font.get_linesize() + 4

        available_h = max(0, control_top - y - 8)
        max_think_lines = max(1, available_h // self.thinking_font.get_linesize())
        latest_thinking = self.thinking_history[-1] if self.thinking_history else "暂无"
        self._draw_wrapped_text(
            text=latest_thinking,
            font=self.thinking_font,
            color=COLORS["text_dark"],
            x=x,
            y=y,
            max_width=max_w,
            max_lines=max_think_lines,
        )

        cy = control_top
        for line in controls:
            control_text = self.info_font.render(line, True, COLORS["text_dark"])
            self.screen.blit(control_text, (x, cy))
            cy += control_line_h

    def update_display(self):
        """更新显示"""
        self.screen.fill((250, 248, 239))  # 背景色
        self.draw_grid()
        self.draw_info()
        pygame.display.flip()

    def run(self, max_steps: int = 1000):
        """运行可视化"""
        running = True
        paused = False
        game_over = False

        print("\n" + "=" * 50)
        print("2048 AI 可视化开始")
        print("=" * 50)
        print("控制: SPACE=暂停/继续, R=重新开始, ESC=退出")
        print("=" * 50 + "\n")

        while running and not game_over and self.step_count < max_steps:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        # 重新开始
                        self.game.reset()
                        self.step_count = 0
                        self.thinking_history = []
                        self.action_history = []
                        game_over = False
                        print("\n重新开始游戏\n")

            if paused:
                self.update_display()
                self.clock.tick(30)
                continue

            # 获取当前状态
            state = self.game._get_state()

            action, thinking = self.get_model_action(state)

            # 记录思考过程
            if thinking:
                self.thinking_history.append(thinking)

            # 检查动作是否合法
            valid_actions = self.game.get_valid_actions()
            action_name = ['上', '右', '下', '左'][action] if 0 <= action <= 3 else '未知'

            if action not in valid_actions:
                print(f"  ⚠️  非法动作: {action_name}，使用随机合法动作")
                if valid_actions:
                    action = valid_actions[0]
                    action_name = ['上', '右', '下', '左'][action]

            # 记录动作
            self.action_history.append(action_name)

            # 执行动作
            _, _, done, _ = self.game.step(action)

            self.step_count += 1

            # 更新显示
            self.update_display()

            # 延迟
            time.sleep(self.delay)

            if done:
                game_over = True

        # 游戏结束
        print("\n" + "=" * 50)
        print("游戏结束")
        print("=" * 50)
        print(f"最终分数: {self.game.score}")
        print(f"最大方块: {self.game.get_max_tile()}")
        print(f"总步数: {self.step_count}")
        print("=" * 50 + "\n")

        # 等待用户关闭
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or \
                   (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    waiting = False
            self.clock.tick(30)

        pygame.quit()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="2048游戏可视化评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 可视化规则智能体（无需模型）
  python scripts/visualize_2048.py --agent_type rule --rule_difficulty advanced

  # 可视化专家策略智能体（expectimax）
  python scripts/visualize_2048.py --agent_type rule --rule_difficulty expert

  # 可视化随机智能体（无需模型）
  python scripts/visualize_2048.py --agent_type random

  # 可视化SFT模型（默认使用vLLM加速）
  python scripts/visualize_2048.py --model_path ./checkpoints/sft

  # 关闭thinking模式（使用非thinking推荐采样参数）
  python scripts/visualize_2048.py --model_path ./checkpoints/sft --no_thinking

  # 不使用vLLM（回退到标准模式）
  python scripts/visualize_2048.py --model_path ./checkpoints/sft --no_vllm

  # 调整速度和窗口大小
  python scripts/visualize_2048.py --model_path ./checkpoints/sft --delay 0.2 --cell_size 80
        """
    )

    parser.add_argument("--agent_type", type=str, default="llm",
                        choices=["llm", "rule", "random"],
                        help="智能体类型：llm/rule/random")
    parser.add_argument("--rule_difficulty", type=str, default="advanced",
                        choices=["basic", "intermediate", "advanced", "expert"],
                        help="规则智能体难度（仅agent_type=rule时有效）")
    parser.add_argument("--model_path", type=str, default="./checkpoints/sft",
                        help="模型路径（SFT后的模型）")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B",
                        help="基座模型名称")
    parser.add_argument("--cell_size", type=int, default=100,
                        help="格子大小（像素）")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="每步延迟（秒）")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="最大步数")
    parser.add_argument("--no_vllm", action="store_true",
                        help="不使用vLLM加速推理")
    parser.add_argument("--vllm_quant", type=str, default="int8",
                        choices=["int8", "awq", "gptq"],
                        help="vLLM量化方式")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    think_group = parser.add_mutually_exclusive_group()
    think_group.add_argument("--use_thinking", dest="use_thinking", action="store_true",
                             help="启用thinking模式（默认开启）")
    think_group.add_argument("--no_thinking", dest="use_thinking", action="store_false",
                             help="关闭thinking模式")
    parser.set_defaults(use_thinking=True)
    parser.add_argument("--temperature", type=float, default=None,
                        help="采样温度覆盖（默认自动: thinking=0.6, non-thinking=0.7）")
    parser.add_argument("--presence_penalty", type=float, default=0.0,
                        help="存在惩罚（仅支持框架生效，建议0~2）")

    args = parser.parse_args()

    if not PYGAME_AVAILABLE:
        raise SystemExit("未安装 pygame，无法打开窗口可视化。请先安装 pygame。")

    if args.agent_type == "llm" and not args.model_path:
        parser.error("agent_type=llm 时需要 --model_path")

    # 创建可视化器
    visualizer = Game2048Visualizer(
        model_path=args.model_path,
        base_model=args.base_model,
        cell_size=args.cell_size,
        delay=args.delay,
        use_vllm=not args.no_vllm,
        vllm_quant=args.vllm_quant,
        agent_type=args.agent_type,
        rule_difficulty=args.rule_difficulty,
        seed=args.seed,
        use_thinking=args.use_thinking,
        temperature=args.temperature,
        presence_penalty=args.presence_penalty,
    )

    # 运行
    visualizer.run(max_steps=args.max_steps)


if __name__ == "__main__":
    main()
