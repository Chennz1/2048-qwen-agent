"""
Data Generator for 2048 Game
Generates training data using various heuristic strategies.
"""

import json
import random
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Literal, Optional, Tuple
import numpy as np

from src.envs.game_2048 import Game2048, ACTION_MAP
from src.data_gen.contracts import SCHEMA_VERSION


# Strategy definitions
DifficultyLevel = Literal['random', 'basic', 'intermediate', 'advanced', 'expert']

DIFFICULTY_LEVELS = {
    'random': 0,
    'basic': 1,
    'intermediate': 2,
    'advanced': 3,
    'expert': 4,
}


class Expectimax2048Policy:
    """Near-optimal 2048 policy via shallow expectimax search."""

    def __init__(self, search_depth: int = 2, max_empty_branches: int = 8):
        self.search_depth = max(1, int(search_depth))
        self.max_empty_branches = max(1, int(max_empty_branches))
        self._cache: Dict[Tuple[bytes, int, str], float] = {}

    @staticmethod
    def _compress_and_merge(line: np.ndarray) -> Tuple[np.ndarray, bool, int]:
        non_zero = line[line > 0]
        original = line.copy()

        merged: List[int] = []
        score_gain = 0
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                val = int(non_zero[i] * 2)
                merged.append(val)
                score_gain += val
                i += 2
            else:
                merged.append(int(non_zero[i]))
                i += 1

        result = np.zeros(4, dtype=np.int32)
        result[:len(merged)] = merged
        moved = not np.array_equal(original, result)
        return result, moved, score_gain

    def _simulate_move(self, grid: np.ndarray, action: int) -> Tuple[np.ndarray, bool, int]:
        new_grid = np.array(grid, dtype=np.int32, copy=True)
        moved = False
        score_gain = 0

        if action == 0:  # up
            for col in range(4):
                line = new_grid[:, col]
                out, line_moved, gain = self._compress_and_merge(line)
                new_grid[:, col] = out
                moved = moved or line_moved
                score_gain += gain
        elif action == 1:  # right
            for row in range(4):
                line = new_grid[row, :][::-1]
                out, line_moved, gain = self._compress_and_merge(line)
                new_grid[row, :] = out[::-1]
                moved = moved or line_moved
                score_gain += gain
        elif action == 2:  # down
            for col in range(4):
                line = new_grid[:, col][::-1]
                out, line_moved, gain = self._compress_and_merge(line)
                new_grid[:, col] = out[::-1]
                moved = moved or line_moved
                score_gain += gain
        elif action == 3:  # left
            for row in range(4):
                line = new_grid[row, :]
                out, line_moved, gain = self._compress_and_merge(line)
                new_grid[row, :] = out
                moved = moved or line_moved
                score_gain += gain

        return new_grid, moved, score_gain

    @staticmethod
    def _max_in_corner_bonus(grid: np.ndarray) -> float:
        max_tile = int(np.max(grid))
        corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
        if any(int(grid[r, c]) == max_tile for r, c in corners):
            return 1.0
        return 0.0

    @staticmethod
    def _smoothness_penalty(grid: np.ndarray) -> float:
        penalty = 0.0
        for r in range(4):
            for c in range(4):
                if grid[r, c] <= 0:
                    continue
                v = float(np.log2(grid[r, c]))
                if r + 1 < 4 and grid[r + 1, c] > 0:
                    penalty += abs(v - float(np.log2(grid[r + 1, c])))
                if c + 1 < 4 and grid[r, c + 1] > 0:
                    penalty += abs(v - float(np.log2(grid[r, c + 1])))
        return penalty

    @staticmethod
    def _monotonicity(grid: np.ndarray) -> float:
        totals = [0.0, 0.0, 0.0, 0.0]  # up down left right
        for r in range(4):
            current = 0.0
            for c in range(3):
                if grid[r, c] == 0 or grid[r, c + 1] == 0:
                    continue
                a = float(np.log2(grid[r, c]))
                b = float(np.log2(grid[r, c + 1]))
                if a >= b:
                    current += 1.0
                else:
                    current -= 1.0
            totals[2] += current
            totals[3] -= current
        for c in range(4):
            current = 0.0
            for r in range(3):
                if grid[r, c] == 0 or grid[r + 1, c] == 0:
                    continue
                a = float(np.log2(grid[r, c]))
                b = float(np.log2(grid[r + 1, c]))
                if a >= b:
                    current += 1.0
                else:
                    current -= 1.0
            totals[0] += current
            totals[1] -= current
        return float(max(totals))

    @staticmethod
    def _merge_potential(grid: np.ndarray) -> int:
        merges = 0
        for r in range(4):
            for c in range(3):
                if grid[r, c] > 0 and grid[r, c] == grid[r, c + 1]:
                    merges += 1
        for r in range(3):
            for c in range(4):
                if grid[r, c] > 0 and grid[r, c] == grid[r + 1, c]:
                    merges += 1
        return merges

    def _evaluate_grid(self, grid: np.ndarray) -> float:
        empty_cells = int(np.sum(grid == 0))
        max_tile = int(np.max(grid))
        sum_log = float(np.sum(np.log2(grid[grid > 0]))) if np.any(grid > 0) else 0.0
        smoothness = self._smoothness_penalty(grid)
        monotonicity = self._monotonicity(grid)
        merge_potential = self._merge_potential(grid)
        corner_bonus = self._max_in_corner_bonus(grid)

        return (
            empty_cells * 300.0
            + monotonicity * 60.0
            - smoothness * 20.0
            + merge_potential * 250.0
            + corner_bonus * 1200.0
            + np.log2(max(max_tile, 2)) * 500.0
            + sum_log * 8.0
        )

    @staticmethod
    def _empty_priority(grid: np.ndarray, rc: Tuple[int, int]) -> float:
        r, c = rc
        score = 0.0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 4 and 0 <= nc < 4 and grid[nr, nc] > 0:
                score += float(np.log2(grid[nr, nc]))
        return score

    def _expectimax_max(self, grid: np.ndarray, depth: int) -> float:
        key = (grid.tobytes(), depth, "max")
        if key in self._cache:
            return self._cache[key]

        if depth <= 0:
            val = self._evaluate_grid(grid)
            self._cache[key] = val
            return val

        best = -float("inf")
        any_move = False
        for action in range(4):
            moved_grid, moved, gain = self._simulate_move(grid, action)
            if not moved:
                continue
            any_move = True
            val = float(gain) + self._expectimax_chance(moved_grid, depth - 1)
            if val > best:
                best = val

        if not any_move:
            best = self._evaluate_grid(grid)
        self._cache[key] = best
        return best

    def _expectimax_chance(self, grid: np.ndarray, depth: int) -> float:
        key = (grid.tobytes(), depth, "chance")
        if key in self._cache:
            return self._cache[key]

        empties = list(zip(*np.where(grid == 0)))
        if depth <= 0 or not empties:
            val = self._evaluate_grid(grid)
            self._cache[key] = val
            return val

        if len(empties) > self.max_empty_branches:
            empties = sorted(empties, key=lambda rc: self._empty_priority(grid, rc), reverse=True)[: self.max_empty_branches]

        cell_prob = 1.0 / len(empties)
        expect = 0.0
        for r, c in empties:
            for tile, p in ((2, 0.9), (4, 0.1)):
                next_grid = np.array(grid, copy=True)
                next_grid[r, c] = tile
                expect += cell_prob * p * self._expectimax_max(next_grid, depth - 1)

        self._cache[key] = expect
        return expect

    def choose_action(self, game: Game2048, valid_actions: List[int]) -> int:
        if not valid_actions:
            return 0

        self._cache.clear()
        grid = np.array(game.grid, dtype=np.int32, copy=True)
        best_action = valid_actions[0]
        best_value = -float("inf")

        for action in valid_actions:
            moved_grid, moved, gain = self._simulate_move(grid, action)
            if not moved:
                continue
            val = float(gain) + self._expectimax_chance(moved_grid, self.search_depth - 1)
            if val > best_value:
                best_value = val
                best_action = action

        return best_action


class CoTDiversityGenerator:
    """
    Chain-of-Thought多样性生成器

    为每个分析部分生成多种表达方式，增加CoT数据的多样性。
    使用独立的随机状态，避免受游戏种子重置的影响。
    """

    # 位置描述模板
    POSITION_TEMPLATES = [
        "最大数字{tile}在{pos}",
        "{tile}位于{pos}",
        "当前最大数字为{tile}，位置在{pos}",
        "棋盘上最大的数字是{tile}，处于{pos}",
        "{pos}有最大数字{tile}",
    ]

    # 角落位置模板
    CORNER_TEMPLATES = [
        "{corner}",
        "位于{corner}",
        "在{corner}位置",
        "占据了{corner}",
    ]

    # 非角落位置模板
    NON_CORNER_TEMPLATES = [
        "在位置{pos}",
        "位于{pos}",
        "不在角落（位置{pos}）",
        "处于{pos}",
    ]

    # 策略描述模板（保持基座）
    STRATEGY_KEEP_TEMPLATES = [
        "选择向{action}移动，保持大数字在{corner}",
        "向{action}移动以维持大数字在{corner}",
        "为了保持大数字在{corner}，选择向{action}移动",
        "向{action}移动，确保大数字安全停在{corner}",
        "优先向{action}展开，保持大数字在{corner}",
    ]

    # 策略描述模板（移动到角落）
    STRATEGY_MOVE_TEMPLATES = [
        "向{action}移动，将大数字移向{corner}",
        "选择向{action}，目标是把大数字移到{corner}",
        "为了将大数字移向{corner}，向{action}移动",
        "向{action}移动，尝试建立{corner}的基座",
    ]

    # 合并机会描述模板
    MERGE_TEMPLATES = [
        "这个方向可以合并相同数字",
        "向{action}移动可以促成合并",
        "有合并机会，向{action}移动",
        "可以合并相同数字，选择向{action}移动",
    ]

    # 无合并机会描述
    NO_MERGE_TEMPLATES = [
        "当前没有明显的合并机会",
        "没有直接的合并机会",
        "暂时没有可以合并的数字",
        "缺少合并机会",
    ]

    # 优先合并描述（为了合并而移出基座）
    PRIORITY_MERGE_TEMPLATES = [
        "虽然会移出{corner}，但向{action}移动可以合并，优先合并",
        "为了合并，选择向{action}移动，虽然会离开{corner}",
        "向{action}移动可以合并，优先考虑合并",
        "虽然有离开{corner}的风险，但为了合并向{action}移动",
    ]

    # 单调性描述模板
    MONOTONICITY_TEMPLATES = [
        "可以改善棋盘单调性",
        "有助于提升棋盘单调性",
        "能让棋盘更有序",
        "可以优化棋盘结构",
    ]

    # 空位描述模板
    SPACE_TEMPLATES = [
        "棋盘较满（剩余{count}格），需要谨慎",
        "只剩{count}个空位，要小心",
        "空间紧张（剩{count}格），谨慎移动",
        "空位不多（{count}格），需要仔细考虑",
    ]

    def __init__(self, seed: int = None):
        """
        初始化多样性生成器

        Args:
            seed: 随机种子（如果提供，使用独立随机状态）
        """
        # 使用独立的随机状态，避免受全局random.seed()影响
        self.rng = random.Random(seed)

    def _random_choice(self, template_list: List[str]) -> str:
        """从模板列表中随机选择一个（使用独立随机状态）"""
        return self.rng.choice(template_list)

    def _format_position(self, pos: tuple[int, int]) -> str:
        """格式化位置（使用独立随机状态）"""
        row_names = ["第一行", "第二行", "第三行", "第四行"]
        col_names = ["第一列", "第二列", "第三列", "第四列"]

        variations = [
            f"({pos[0]+1}, {pos[1]+1})",
            f"{row_names[pos[0]]}{col_names[pos[1]]}",
            f"第{pos[0]+1}行第{pos[1]+1}列",
        ]

        return self.rng.choice(variations)

    def _describe_action(self, action_id: int) -> str:
        """描述动作（多种表达，使用独立随机状态）"""
        action_name = ACTION_MAP[action_id]

        variations = [
            action_name,
            f"向{action_name}",
            f"选择{action_name}",
        ]

        return self.rng.choice(variations)

    def generate_position_description(
        self,
        max_tile: int,
        max_pos: tuple[int, int],
        in_corner: bool,
        corner_name: str = None
    ) -> str:
        """
        生成位置描述（多种表达）

        Args:
            max_tile: 最大数字
            max_pos: 最大数字位置
            in_corner: 是否在角落
            corner_name: 角落名称（如果在角落）
        """
        if in_corner and corner_name:
            # 在角落的情况
            pos_template = self._random_choice(self.CORNER_TEMPLATES)
            position_desc = pos_template.format(corner=corner_name)

            base_templates = self.POSITION_TEMPLATES
            base_template = self._random_choice(base_templates)

            # 组合
            variations = [
                base_template.format(tile=max_tile, pos=position_desc),
                f"{max_tile}{position_desc}",
                f"{corner_name}有{max_tile}",
            ]
            return self._random_choice(variations)
        else:
            # 不在角落的情况
            pos_desc = self._format_position(max_pos)

            base_templates = self.POSITION_TEMPLATES
            base_template = self._random_choice(base_templates)

            variations = [
                base_template.format(tile=max_tile, pos=pos_desc),
                f"{max_tile}{pos_desc}",
            ]
            return self._random_choice(variations)

    def generate_strategy_description(
        self,
        action_id: int,
        action_name: str,
        max_tile: int,
        max_pos: tuple[int, int],
        in_corner: bool,
        corner_name: str = None,
        preserves_corner: bool = True
    ) -> str:
        """
        生成策略描述（多种表达）

        Args:
            action_id: 动作ID
            action_name: 动作名称
            max_tile: 最大数字
            max_pos: 最大数字位置
            in_corner: 是否在角落
            corner_name: 角落名称
            preserves_corner: 是否保持基座
        """
        if in_corner and preserves_corner:
            # 保持基座的策略
            templates = self.STRATEGY_KEEP_TEMPLATES
            template = self._random_choice(templates)
            return template.format(action=action_name, corner=corner_name)
        else:
            # 移动到角落的策略
            if corner_name:
                templates = self.STRATEGY_MOVE_TEMPLATES
                template = self._random_choice(templates)
                return template.format(action=action_name, corner=corner_name)
            else:
                return f"选择向{action_name}移动"

    def generate_merge_description(
        self,
        has_merge: bool,
        action_name: str = None,
        corner_name: str = None
    ) -> str:
        """
        生成合并机会描述（多种表达）

        Args:
            has_merge: 是否有合并机会
            action_name: 动作名称
            corner_name: 当前基座角落（如果需要）
        """
        if has_merge:
            templates = self.MERGE_TEMPLATES
            template = self._random_choice(templates)

            if "{action}" in template:
                return template.format(action=action_name if action_name else "")
            return template
        else:
            templates = self.NO_MERGE_TEMPLATES
            return self._random_choice(templates)

    def generate_priority_merge_description(
        self,
        action_name: str,
        corner_name: str
    ) -> str:
        """生成优先合并描述（为了合并移出基座）"""
        templates = self.PRIORITY_MERGE_TEMPLATES
        template = self._random_choice(templates)
        return template.format(action=action_name, corner=corner_name)

    def generate_monotonicity_description(self) -> str:
        """生成单调性描述（多种表达）"""
        return self._random_choice(self.MONOTONICITY_TEMPLATES)

    def generate_space_description(self, empty_count: int) -> str:
        """生成空位描述（多种表达）"""
        if empty_count < 4:
            templates = self.SPACE_TEMPLATES
            template = self._random_choice(templates)
            return template.format(count=empty_count)
        return ""



class ThinkingHeuristicPlayer:
    """
    启发式2048玩家，带Chain-of-Thought思考过程

    与普通HeuristicPlayer的区别：
    - 每次选择动作时会生成真实的思考过程
    - 思考过程基于对棋盘的实际分析
    - 思考内容包含：最大数字位置、合并机会、移动后果等
    - 支持CoT多样性生成，避免单一表达
    """

    SNAPSHOT_TEMPLATES = (
        "最大数字{max_tile}，空位{empty_cells}个，可行动作：{valid_actions}",
        "当前最大数字{max_tile}，空位{empty_cells}个，可行动作：{valid_actions}",
        "棋盘关键信息：最大数字{max_tile}，空位{empty_cells}个，可行动作：{valid_actions}",
    )
    CORNER_KEEP_TEMPLATES = (
        "最大数字{max_tile}在{corner}，该动作可以保持角落基座",
        "最大数字{max_tile}位于{corner}，该方向有助于稳住角落基座",
        "最大数字{max_tile}在{corner}，该走法能维持角落锚点",
    )
    CORNER_BREAK_TEMPLATES = (
        "最大数字{max_tile}在{corner}，该动作会牺牲部分角落稳定性",
        "最大数字{max_tile}位于{corner}，该走法会削弱角落控制",
        "最大数字{max_tile}在{corner}，该方向存在破坏角落基座的风险",
    )
    NON_CORNER_TEMPLATES = (
        "最大数字{max_tile}不在角落，优先把大数字往角落方向整理",
        "最大数字{max_tile}尚未入角，优先向角落组织棋形",
        "最大数字{max_tile}不在角落，先建立角落基座更稳妥",
    )
    MERGE_GAIN_TEMPLATES = (
        "向{action_name}移动可立即产生合并并提升分数",
        "向{action_name}走可直接触发合并，短期收益更高",
        "选择{action_name}可形成即时合并，回合价值更优",
    )
    MERGE_SETUP_TEMPLATES = (
        "向{action_name}移动短期不一定合并，但有利于后续结构",
        "向{action_name}走当前未必立刻合并，但能为后续合并铺路",
        "选择{action_name}虽无即时合并，但可优化下一步连锁机会",
    )
    MONOTONICITY_TEMPLATES = (
        "该动作可以提升棋盘单调性",
        "该走法有助于增强棋盘单调结构",
        "这个方向能让棋盘更接近单调布局",
    )
    LOW_SPACE_TEMPLATES = (
        "当前空位仅{empty_count}个，需要优先避免死局",
        "空位只剩{empty_count}个，需优先规避卡死风险",
        "当前仅有{empty_count}个空位，先保证生存空间更重要",
    )
    FINAL_DECISION_TEMPLATES = (
        "最终选择向{action_name}移动",
        "最终决定向{action_name}移动",
        "综上，选择向{action_name}移动",
    )

    def __init__(
        self,
        difficulty: DifficultyLevel = 'intermediate',
        seed: int = None,
        enable_diversity: bool = True,
        expert_depth: int = 2,
        expert_max_empty: int = 8,
    ):
        """
        初始化启发式玩家

        Args:
            difficulty: 难度等级 ('random', 'basic', 'intermediate', 'advanced', 'expert')
            seed: 随机种子
            enable_diversity: 兼容参数（当前版本始终启用CoT多样性）
            expert_depth: expert策略expectimax搜索深度
            expert_max_empty: expert策略chance节点最大空位分支数
        """
        self.difficulty = difficulty
        self.enable_diversity = True
        self.diversity_generator = CoTDiversityGenerator(seed=seed)
        # Dedicated RNG for CoT phrasing diversity.
        self._thinking_rng = random.Random(seed)
        self.expert_policy: Optional[Expectimax2048Policy] = None
        if self.difficulty == 'expert':
            self.expert_policy = Expectimax2048Policy(
                search_depth=expert_depth,
                max_empty_branches=expert_max_empty,
            )

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _pick_template(self, templates: tuple[str, ...]) -> str:
        return self._thinking_rng.choice(templates)

    def choose_action_with_thinking(self, game: Game2048) -> tuple[str, int]:
        """
        选择动作并生成思考过程

        重要：思考过程基于实际选择的动作，确保一致性

        Args:
            game: 当前游戏状态

        Returns:
            (thinking_text, action_id) - 思考文本和动作ID
        """
        # 先选择动作
        action = self.choose_action(game)

        # 基于选择的动作生成思考过程
        thinking = self._analyze_board_for_action(game, action)

        return thinking, action

    def _analyze_board_for_action(self, game: Game2048, chosen_action: int) -> str:
        """
        为选定动作生成思考过程。

        Args:
            game: 游戏状态
            chosen_action: 已经选择的动作

        Returns:
            思考文本
        """
        return self._analyze_with_diversity(game, chosen_action)

    def _build_state_snapshot(self, game: Game2048) -> str:
        """Build a concise, high-information board summary for CoT."""
        valid_actions = game.get_valid_actions()
        valid_actions_text = "、".join(ACTION_MAP[a] for a in valid_actions) if valid_actions else "无"
        template = self._pick_template(self.SNAPSHOT_TEMPLATES)
        return template.format(
            max_tile=game.get_max_tile(),
            empty_cells=game.get_empty_cells(),
            valid_actions=valid_actions_text,
        )

    def _finalize_thinking(self, parts: List[str], chosen_action: int) -> str:
        """Normalize CoT text with punctuation and explicit final decision."""
        normalized: List[str] = []
        for part in parts:
            text = str(part).strip()
            if not text:
                continue
            text = text.rstrip("；。,. ")
            if text and text not in normalized:
                normalized.append(text)

        action_name = ACTION_MAP[chosen_action]
        final_template = self._pick_template(self.FINAL_DECISION_TEMPLATES)
        normalized.append(final_template.format(action_name=action_name))
        return "；".join(normalized) + "。"

    def _analyze_with_diversity(self, game: Game2048, chosen_action: int) -> str:
        """
        使用多样性生成器分析棋盘

        Args:
            game: 游戏状态
            chosen_action: 已选择的动作

        Returns:
            思考文本
        """
        parts = [self._build_state_snapshot(game)]
        gen = self.diversity_generator

        # 1. 分析最大数字位置
        max_tile = game.get_max_tile()
        max_pos = self._find_max_tile_position(game)

        corner_names = {
            (0, 0): "左上角",
            (0, 3): "右上角",
            (3, 0): "左下角",
            (3, 3): "右下角"
        }

        action_name = ACTION_MAP[chosen_action]

        if max_pos in corner_names:
            corner_name = corner_names[max_pos]
            in_corner = True

            # 生成位置描述
            position_desc = gen.generate_position_description(
                max_tile, max_pos, in_corner, corner_name
            )
            parts.append(position_desc)

            # 检查选定的动作是否有利于保持基座
            preserves_corner = self._action_preserves_corner(max_pos, chosen_action)

            if preserves_corner:
                # 保持基座的策略
                strategy_desc = gen.generate_strategy_description(
                    chosen_action, action_name, max_tile, max_pos, in_corner, corner_name, preserves_corner=True
                )
                parts.append(strategy_desc)
            else:
                # 为了合并移出基座
                merge_opportunities = self._find_merge_opportunities(game)
                if chosen_action in merge_opportunities:
                    priority_desc = gen.generate_priority_merge_description(action_name, corner_name)
                    parts.append(priority_desc)
                else:
                    parts.append(f"选择向{action_name}移动。")

            # 分析合并机会
            merge_opportunities = self._find_merge_opportunities(game)
            if chosen_action in merge_opportunities:
                merge_desc = gen.generate_merge_description(
                    has_merge=True, action_name=action_name
                )
                parts.append(merge_desc)
            else:
                merge_desc = gen.generate_merge_description(has_merge=False)
                parts.append(merge_desc)

        else:
            # 不在角落
            in_corner = False
            parts.append(gen.generate_position_description(
                max_tile, max_pos, in_corner
            ))

            # 分析向角落移动
            nearest_corner = self._find_nearest_corner(max_pos)
            if nearest_corner:
                corner_name = corner_names[nearest_corner]
                suggested = self._suggest_action_to_corner(max_pos, nearest_corner)
                if suggested == action_name:
                    strategy_desc = gen.generate_strategy_description(
                        chosen_action, action_name, max_tile, max_pos, in_corner, corner_name, preserves_corner=False
                    )
                    parts.append(strategy_desc)
                else:
                    parts.append(f"选择向{action_name}移动。")

            # 分析合并机会
            merge_opportunities = self._find_merge_opportunities(game)
            if chosen_action in merge_opportunities:
                merge_desc = gen.generate_merge_description(
                    has_merge=True, action_name=action_name
                )
                parts.append(merge_desc)

        # 3. 根据难度添加额外分析
        if self.difficulty in ['intermediate', 'advanced', 'expert']:
            # 检查单调性
            test_game = game.clone()
            test_game._move(chosen_action)
            new_monotonicity = self._calculate_monotonicity(test_game.grid)
            old_monotonicity = self._calculate_monotonicity(game.grid)

            if new_monotonicity > old_monotonicity:
                mono_desc = gen.generate_monotonicity_description()
                parts.append(mono_desc)

        if self.difficulty in ['advanced', 'expert']:
            # 空位情况
            empty_count = game.get_empty_cells()
            if empty_count < 4:
                space_desc = gen.generate_space_description(empty_count)
                if space_desc:
                    parts.append(space_desc)

        return self._finalize_thinking(parts, chosen_action)

    def _action_preserves_corner(self, corner_pos: tuple[int, int], action: int) -> bool:
        """
        检查动作是否有利于保持基座

        Args:
            corner_pos: (row, col) 基座位置
            action: 动作ID (0=上, 1=右, 2=下, 3=左)

        Returns:
            是否保持基座
        """
        row, col = corner_pos

        # 根据基座位置，判断哪些动作会移出基座
        if row == 0:  # 上方两角
            if action == 2:  # 下移会把大数字移出上方
                return False
        else:  # 下方两角
            if action == 0:  # 上移会把大数字移出下方
                return False

        if col == 0:  # 左方两角
            if action == 1:  # 右移会把大数字移出左方
                return False
        else:  # 右方两角
            if action == 3:  # 左移会把大数字移出右方
                return False

        return True

    def choose_action(self, game: Game2048) -> int:
        """
        Choose the best action based on the difficulty level.

        Args:
            game: Current game state

        Returns:
            Action ID (0-3)
        """
        valid_actions = game.get_valid_actions()

        if not valid_actions:
            return 0

        if self.difficulty == 'random':
            return self._random_action(valid_actions)
        elif self.difficulty == 'basic':
            return self._basic_action(game, valid_actions)
        elif self.difficulty == 'intermediate':
            return self._intermediate_action(game, valid_actions)
        elif self.difficulty == 'advanced':
            return self._advanced_action(game, valid_actions)
        else:  # expert
            return self._expert_action(game, valid_actions)

    def _random_action(self, valid_actions: List[int]) -> int:
        """Random action"""
        return random.choice(valid_actions)

    def _expert_action(self, game: Game2048, valid_actions: List[int]) -> int:
        """Expert strategy: expectimax search with stochastic spawn modeling."""
        if self.expert_policy is None:
            self.expert_policy = Expectimax2048Policy(search_depth=2, max_empty_branches=8)
        return self.expert_policy.choose_action(game, valid_actions)

    # ===== 思考过程相关方法 =====

    def _analyze_board(self, game: Game2048) -> str:
        """
        分析棋盘状态，生成思考过程

        这是CoT的核心 - 基于真实棋盘状态进行分析

        Returns:
            思考文本
        """
        parts = []

        # 1. 分析最大数字位置和基座选择
        max_tile = game.get_max_tile()
        max_pos = self._find_max_tile_position(game)

        # 获取角的位置名称
        corner_names = {
            (0, 0): "左上角",
            (0, 3): "右上角",
            (3, 0): "左下角",
            (3, 3): "右下角"
        }

        if max_pos in corner_names:
            corner_name = corner_names[max_pos]
            parts.append(f"最大数字{max_tile}在{corner_name}。")

            # 分析应该保持哪个基座
            preferred_direction = self._get_preferred_direction_for_corner(max_pos)
            if preferred_direction:
                parts.append(f"应该保持大数字在{corner_name}，优先向{preferred_direction}方向展开。")
        else:
            parts.append(f"最大数字{max_tile}在位置{max_pos}，不在角落。")

            # 建议移动到最近的角
            nearest_corner = self._find_nearest_corner(max_pos)
            if nearest_corner:
                corner_name = corner_names[nearest_corner]
                suggested_action = self._suggest_action_to_corner(max_pos, nearest_corner)
                if suggested_action:
                    parts.append(f"应该向{suggested_action}移动，将大数字移向{corner_name}。")

        # 2. 分析合并机会
        merge_opportunities = self._find_merge_opportunities(game)
        if merge_opportunities:
            if len(merge_opportunities) == 1:
                action_name = ACTION_MAP[merge_opportunities[0]]
                parts.append(f"向{action_name}移动可以合并。")
            else:
                actions_str = "、".join([ACTION_MAP[a] for a in merge_opportunities])
                parts.append(f"可以向{actions_str}移动合并。")
        else:
            parts.append("当前没有合并机会。")

        # 3. 根据难度添加不同深度的分析
        if self.difficulty in ['intermediate', 'advanced', 'expert']:
            # 分析空位情况
            empty_count = game.get_empty_cells()
            if empty_count < 4:
                parts.append(f"棋盘较满（剩余{empty_count}格），需要谨慎。")

            # 分析单调性
            if self.difficulty in ['advanced', 'expert']:
                monotonicity = self._calculate_monotonicity(game.grid)
                if monotonicity > 3:
                    parts.append("棋盘单调性好，继续保持。")
                elif monotonicity < 0:
                    parts.append("棋盘单调性差，需要调整。")

        return "".join(parts)

    def _get_preferred_direction_for_corner(self, corner_pos: tuple[int, int]) -> str:
        """
        根据基座位置，返回优先展开的方向

        Args:
            corner_pos: (row, col) 角落位置

        Returns:
            优先方向描述
        """
        row, col = corner_pos

        # 根据基座位置，建议不要把大数字移出这个角
        if row == 0:  # 上方两角
            if col == 0:  # 左上角
                return "右、下"  # 可以向右或向下，不要向上或向左
            else:  # 右上角
                return "左、下"  # 可以向左或向下，不要向上或向右
        else:  # 下方两角
            if col == 0:  # 左下角
                return "右、上"  # 可以向右或向上，不要向下或向左
            else:  # 右下角
                return "左、上"  # 可以向左或向上，不要向下或向右

    def _find_nearest_corner(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        """找到最近的角落"""
        row, col = pos
        corners = [(0, 0), (0, 3), (3, 0), (3, 3)]

        nearest = None
        min_dist = float('inf')

        for corner in corners:
            dist = abs(row - corner[0]) + abs(col - corner[1])
            if dist < min_dist:
                min_dist = dist
                nearest = corner

        return nearest

    def _suggest_action_to_corner(self, pos: tuple[int, int], corner: tuple[int, int]) -> str | None:
        """建议从当前位置移动到指定角落的动作"""
        row, col = pos
        corner_row, corner_col = corner

        # 先处理行方向
        if row < corner_row:
            return "下"
        elif row > corner_row:
            return "上"

        # 再处理列方向
        if col < corner_col:
            return "右"
        elif col > corner_col:
            return "左"

        # 已经在角落了
        return None

    def _find_max_tile_position(self, game: Game2048) -> tuple[int, int]:
        """找到最大数字的位置"""
        max_tile = game.get_max_tile()
        for i in range(4):
            for j in range(4):
                if game.grid[i, j] == max_tile:
                    return (i, j)
        return (0, 0)

    def _find_merge_opportunities(self, game: Game2048) -> List[int]:
        """
        找到可以合并的移动方向

        Returns:
            可以合并的动作列表
        """
        merge_actions = []

        for action in range(4):
            test_game = game.clone()
            prev_score = test_game.score
            test_game._move(action)
            score_gain = test_game.score - prev_score

            # 如果移动后有得分增长，说明有合并
            if score_gain > 0:
                merge_actions.append(action)

        return merge_actions

    # ===== 原有的评估方法 =====

    def _basic_action(self, game: Game2048, valid_actions: List[int]) -> int:
        """
        Basic strategy: Prioritize moves that increase score
        """
        best_action = valid_actions[0]
        best_score_gain = -float('inf')

        for action in valid_actions:
            # Simulate the move
            test_game = game.clone()
            prev_score = test_game.score
            test_game._move(action)
            score_gain = test_game.score - prev_score

            if score_gain > best_score_gain:
                best_score_gain = score_gain
                best_action = action

        return best_action

    def _intermediate_action(self, game: Game2048, valid_actions: List[int]) -> int:
        """
        Intermediate strategy: Consider score and empty cells
        - Prioritize merging tiles (score gain)
        - Prefer moves that keep large tiles in corner
        - Avoid moves that reduce empty cells significantly
        """
        best_action = valid_actions[0]
        best_score = -float('inf')

        for action in valid_actions:
            score = self._evaluate_move_intermediate(game, action)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _evaluate_move_intermediate(self, game: Game2048, action: int) -> float:
        """Evaluate a move for intermediate strategy"""
        test_game = game.clone()
        prev_score = test_game.score
        prev_empty = test_game.get_empty_cells()

        test_game._move(action)
        score_gain = test_game.score - prev_score
        new_empty = test_game.get_empty_cells()

        # Score components
        empty_bonus = (new_empty - prev_empty) * 10
        monotonicity_bonus = self._calculate_monotonicity(test_game.grid) * 5

        return score_gain + empty_bonus + monotonicity_bonus

    def _advanced_action(self, game: Game2048, valid_actions: List[int]) -> int:
        """
        Advanced strategy: Full evaluation including:
        - Snake pattern for monotonicity
        - Corner strategy (keep max tile in corner)
        - Merge potential
        - Empty cells preservation
        """
        best_action = valid_actions[0]
        best_score = -float('inf')

        for action in valid_actions:
            score = self._evaluate_move_advanced(game, action)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _evaluate_move_advanced(self, game: Game2048, action: int) -> float:
        """Evaluate a move for advanced strategy"""
        test_game = game.clone()
        prev_score = test_game.score
        prev_max = test_game.get_max_tile()

        test_game._move(action)
        score_gain = test_game.score - prev_score
        new_max = test_game.get_max_tile()

        # Score components
        empty_bonus = test_game.get_empty_cells() * 15
        corner_bonus = self._corner_score(test_game.grid) * 20
        monotonicity = self._calculate_monotonicity(test_game.grid) * 10
        smoothness = self._calculate_smoothness(test_game.grid) * 5
        max_tile_bonus = (new_max - prev_max) * 2

        total = (
            score_gain * 2 +
            empty_bonus +
            corner_bonus +
            monotonicity +
            smoothness +
            max_tile_bonus
        )

        return total

    def _corner_score(self, grid: np.ndarray) -> float:
        """
        Calculate corner strategy score.
        Prefer keeping the max tile in a corner.
        """
        max_tile = np.max(grid)
        corners = [(0, 0), (0, 3), (3, 0), (3, 3)]

        for i, j in corners:
            if grid[i, j] == max_tile:
                return 1.0

        # Check if max tile is adjacent to corner
        for i, j in corners:
            neighbors = []
            if i > 0:
                neighbors.append(grid[i - 1, j])
            if i < 3:
                neighbors.append(grid[i + 1, j])
            if j > 0:
                neighbors.append(grid[i, j - 1])
            if j < 3:
                neighbors.append(grid[i, j + 1])

            if max_tile in neighbors:
                return 0.5

        return 0.0

    def _calculate_monotonicity(self, grid: np.ndarray) -> float:
        """
        Calculate monotonicity score.
        Prefer tiles arranged in increasing/decreasing order.
        Empty tiles (value 0) are skipped to avoid false positives.
        """
        totals = [0.0, 0.0, 0.0, 0.0]  # up, down, left, right

        # Left/Right
        for i in range(4):
            current = 0
            for j in range(3):
                # Skip empty tiles in comparison
                if grid[i, j] == 0 or grid[i, j + 1] == 0:
                    continue
                if grid[i, j] >= grid[i, j + 1]:
                    current += 1
                elif grid[i, j] < grid[i, j + 1]:
                    current -= 1
            totals[2] += current  # left
            totals[3] -= current  # right

        # Up/Down
        for j in range(4):
            current = 0
            for i in range(3):
                # Skip empty tiles in comparison
                if grid[i, j] == 0 or grid[i + 1, j] == 0:
                    continue
                if grid[i, j] >= grid[i + 1, j]:
                    current += 1
                elif grid[i, j] < grid[i + 1, j]:
                    current -= 1
            totals[0] += current  # up
            totals[1] -= current  # down

        return max(totals)

    def _calculate_smoothness(self, grid: np.ndarray) -> float:
        """
        Calculate smoothness score.
        Prefer adjacent tiles having similar values.
        """
        smoothness = 0.0

        for i in range(4):
            for j in range(4):
                if grid[i, j] > 0:
                    value = np.log2(grid[i, j])
                    # Check right neighbor
                    if j < 3 and grid[i, j + 1] > 0:
                        neighbor_value = np.log2(grid[i, j + 1])
                        smoothness -= abs(value - neighbor_value)
                    # Check down neighbor
                    if i < 3 and grid[i + 1, j] > 0:
                        neighbor_value = np.log2(grid[i + 1, j])
                        smoothness -= abs(value - neighbor_value)

        return smoothness


def generate_games(
    num_games: int = 10000,
    max_steps: int = 10000,
    difficulty: DifficultyLevel = 'intermediate',
    seed: int = None,
    with_thinking: bool = False,
    enable_diversity: bool = True,
    expert_depth: int = 2,
    expert_max_empty: int = 8,
) -> List[Dict]:
    """
    Generate training games.

    Args:
        num_games: Number of games to generate
        max_steps: Maximum steps per game
        difficulty: Difficulty level for the heuristic player
        seed: Random seed (if provided, each game will use seed + game_idx)
        with_thinking: Whether to include Chain-of-Thought reasoning
        enable_diversity: Compatibility flag (CoT diversity is always enabled)
        expert_depth: expectimax搜索深度（仅expert难度生效）
        expert_max_empty: chance节点最大空位分支数（仅expert难度生效）

    Returns:
        List of games, each containing states, actions, final_score, max_tile
        If with_thinking=True, each state will also include 'thinking' field
    """
    # 只在初始时设置全局种子，player不设置种子以保持随机性
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 根据with_thinking选择玩家类型
    if with_thinking:
        player = ThinkingHeuristicPlayer(
            difficulty=difficulty,
            seed=None,
            enable_diversity=enable_diversity,
            expert_depth=expert_depth,
            expert_max_empty=expert_max_empty,
        )
    else:
        player = ThinkingHeuristicPlayer(
            difficulty=difficulty,
            seed=None,
            enable_diversity=enable_diversity,
            expert_depth=expert_depth,
            expert_max_empty=expert_max_empty,
        )

    games = []

    for game_idx in range(num_games):
        # 每个游戏使用独立的种子，基于初始种子
        game_seed = seed + game_idx if seed is not None else None
        game = Game2048(seed=game_seed)
        states = []
        actions = []
        scores = []
        steps = []
        thinkings = [] if with_thinking else None

        for step in range(max_steps):
            state = game._get_state()

            if with_thinking:
                # 使用带思考的动作选择
                thinking, action = player.choose_action_with_thinking(game)
                thinkings.append(thinking)
            else:
                # 普通动作选择
                action = player.choose_action(game)

            states.append(state)
            actions.append(action)
            scores.append(game.score)
            steps.append(step)

            _, _, done, _ = game.step(action)

            if done:
                break

        # 构建游戏数据
        game_data = {
            'schema_version': SCHEMA_VERSION,
            'game_id': f"game_{game_idx:06d}",
            'difficulty': difficulty,
            'states': [],
            'final_score': game.score,
            'max_tile': game.get_max_tile(),
            'total_steps': len(states)
        }

        # 添加每个状态的数据
        for i in range(len(states)):
            state_data = {
                'state': states[i],
                'action': ACTION_MAP[actions[i]],
                'action_id': actions[i],
                'score': scores[i],
                'step': steps[i]
            }

            # 如果有思考过程，添加thinking字段
            if with_thinking:
                state_data['thinking'] = thinkings[i]

            game_data['states'].append(state_data)

        games.append(game_data)

        if (game_idx + 1) % 1000 == 0:
            print(f"Generated {game_idx + 1}/{num_games} games")

    return games


def generate_mixed_data(
    total_games: int = 100000,
    seed: int = None,
    with_thinking: bool = False,
    enable_diversity: bool = True,
    expert_depth: int = 2,
    expert_max_empty: int = 8,
) -> List[Dict]:
    """
    Generate mixed difficulty data.

    Args:
        total_games: Total number of games to generate
        seed: Random seed (base seed, each difficulty level uses different offset)
        with_thinking: Whether to include Chain-of-Thought reasoning
        enable_diversity: Compatibility flag (CoT diversity is always enabled)
        expert_depth: expectimax搜索深度（mixed中若包含expert时生效）
        expert_max_empty: chance节点最大空位分支数

    Returns:
        List of games with mixed difficulty levels
    """
    distribution = {
        'random': 0.1,       # 10% random
        'basic': 0.3,        # 30% basic
        'intermediate': 0.4, # 40% intermediate
        'advanced': 0.2      # 20% advanced
    }

    all_games = []
    seed_offset = 0

    for level, ratio in distribution.items():
        n = int(total_games * ratio)
        # 使用不同的种子偏移，避免重复
        level_seed = seed + seed_offset if seed is not None else None
        print(f"Generating {n} {level} games...")
        games = generate_games(
            num_games=n,
            max_steps=10000,
            difficulty=level,
            seed=level_seed,
            with_thinking=with_thinking,
            enable_diversity=enable_diversity,
            expert_depth=expert_depth,
            expert_max_empty=expert_max_empty,
        )
        all_games.extend(games)
        seed_offset += n  # 增加偏移量

    return all_games


def filter_games_by_quality(
    games: List[Dict],
    min_final_score: int = 0,
    min_steps: int = 1,
    min_unique_state_ratio: float = 0.0,
) -> List[Dict]:
    """
    Lightweight quality gate for beginner-friendly data generation.

    Args:
        games: Raw generated games
        min_final_score: Drop games below this final score
        min_steps: Drop very short games
        min_unique_state_ratio: Drop highly repetitive trajectories

    Returns:
        Filtered games
    """
    filtered: List[Dict] = []

    for game in games:
        final_score = int(game.get("final_score", 0))
        total_steps = int(game.get("total_steps", 0))
        states = game.get("states", [])

        if final_score < min_final_score:
            continue
        if total_steps < min_steps:
            continue

        if states and min_unique_state_ratio > 0:
            unique_states = len({item.get("state", "") for item in states})
            unique_ratio = unique_states / max(len(states), 1)
            if unique_ratio < min_unique_state_ratio:
                continue

        filtered.append(game)

    return filtered


def build_data_quality_report(games: List[Dict]) -> Dict:
    """Build lightweight quality metrics for generated dataset."""
    if not games:
        return {
            "num_games": 0,
            "num_steps": 0,
            "mean_final_score": 0.0,
            "max_final_score": 0,
            "min_final_score": 0,
            "mean_max_tile": 0.0,
            "max_tile_reached": 0,
            "tile_2048_games": 0,
            "difficulty_distribution": {},
            "action_distribution": {},
            "overall_unique_state_ratio": 0.0,
            "mean_unique_state_ratio_per_game": 0.0,
            "cot_step_coverage": 0.0,
            "mean_thinking_length_chars": 0.0,
            "high_info_cot_ratio": 0.0,
        }

    final_scores = [int(g["final_score"]) for g in games]
    max_tiles = [int(g["max_tile"]) for g in games]
    difficulty_counter: Counter = Counter()
    action_counter: Counter = Counter()
    total_states = 0
    unique_state_pool = set()
    per_game_unique_ratios = []
    total_thinking_steps = 0
    total_thinking_chars = 0
    high_info_cot_steps = 0
    info_keywords = ["最大数字", "空位", "可行动作", "最终选择"]

    for game in games:
        difficulty_counter[str(game.get("difficulty", "unknown"))] += 1
        states = game.get("states", [])
        state_texts = []
        for item in states:
            action_counter[str(item.get("action", ""))] += 1
            state = item.get("state", "")
            state_texts.append(state)
            unique_state_pool.add(state)
            thinking = item.get("thinking")
            if isinstance(thinking, str) and thinking.strip():
                total_thinking_steps += 1
                total_thinking_chars += len(thinking.strip())
                keyword_hits = sum(1 for kw in info_keywords if kw in thinking)
                if keyword_hits >= 2:
                    high_info_cot_steps += 1

        total_states += len(state_texts)
        if state_texts:
            per_game_unique_ratios.append(len(set(state_texts)) / len(state_texts))

    return {
        "num_games": len(games),
        "num_steps": int(total_states),
        "mean_final_score": float(np.mean(final_scores)),
        "max_final_score": int(np.max(final_scores)),
        "min_final_score": int(np.min(final_scores)),
        "mean_max_tile": float(np.mean(max_tiles)),
        "max_tile_reached": int(np.max(max_tiles)),
        "tile_2048_games": int(sum(1 for t in max_tiles if t >= 2048)),
        "difficulty_distribution": dict(sorted(difficulty_counter.items())),
        "action_distribution": dict(sorted(action_counter.items())),
        "overall_unique_state_ratio": float(len(unique_state_pool) / max(total_states, 1)),
        "mean_unique_state_ratio_per_game": float(np.mean(per_game_unique_ratios)) if per_game_unique_ratios else 0.0,
        "cot_step_coverage": float(total_thinking_steps / max(total_states, 1)),
        "mean_thinking_length_chars": float(total_thinking_chars / max(total_thinking_steps, 1)),
        "high_info_cot_ratio": float(high_info_cot_steps / max(total_thinking_steps, 1)),
    }


def build_generation_manifest(
    *,
    args: argparse.Namespace,
    report: Dict,
    output_dir: str,
) -> Dict:
    """Build manifest metadata for generated raw data versioning."""
    manifest = {
        "manifest_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "raw_generation",
        "schema_version": SCHEMA_VERSION,
        "output_dir": output_dir,
        "config": {
            "num_games": args.num_games,
            "difficulty": args.difficulty,
            "seed": args.seed,
            "with_thinking": args.with_thinking,
            "enable_diversity": args.enable_diversity,
            "expert_depth": getattr(args, "expert_depth", 2),
            "expert_max_empty": getattr(args, "expert_max_empty", 8),
            "min_final_score": args.min_final_score,
            "min_steps": args.min_steps,
            "min_unique_state_ratio": args.min_unique_state_ratio,
        },
        "report": report,
    }
    return manifest


def _convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_games(games: List[Dict], output_dir: str = "data/raw") -> None:
    """
    Save generated games to JSON files.

    Args:
        games: List of game dictionaries
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for game in games:
        filepath = output_path / f"{game['game_id']}.json"
        # Convert numpy types to Python native types
        game_serializable = _convert_numpy_types(game)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(game_serializable, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(games)} games to {output_dir}")


def main():
    """Main entry point for data generation"""
    parser = argparse.ArgumentParser(description='Generate 2048 training data')
    parser.add_argument('--num_games', type=int, default=10000,
                        help='Number of games to generate')
    parser.add_argument('--difficulty', type=str, default='intermediate',
                        choices=['random', 'basic', 'intermediate', 'advanced', 'expert', 'mixed'],
                        help='Difficulty level')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--with_thinking', action='store_true',
                        help='包含Chain-of-Thought思考过程')
    parser.add_argument('--enable_diversity', action='store_true', default=True,
                        help='兼容参数：CoT多样性当前固定启用')
    parser.add_argument('--no_enable_diversity', dest='enable_diversity', action='store_false',
                        help='兼容参数（已忽略）：当前版本不支持关闭CoT多样性')
    parser.add_argument('--expert_depth', type=int, default=2,
                        help='expert策略expectimax搜索深度（默认2）')
    parser.add_argument('--expert_max_empty', type=int, default=8,
                        help='expert策略chance节点最大空位分支数（默认8）')
    parser.add_argument('--min_final_score', type=int, default=0,
                        help='过滤低质量对局：最低最终分数（默认0，不过滤）')
    parser.add_argument('--min_steps', type=int, default=1,
                        help='过滤低质量对局：最少步数（默认1）')
    parser.add_argument('--min_unique_state_ratio', type=float, default=0.0,
                        help='过滤高重复轨迹：每局最小唯一状态比例（0~1，默认0）')
    parser.add_argument('--report_file', type=str, default=None,
                        help='可选：将数据质量报告保存为JSON文件')
    parser.add_argument('--manifest_file', type=str, default=None,
                        help='可选：将数据版本manifest保存为JSON文件（默认 output_dir/manifest.json）')

    args = parser.parse_args()
    if not args.enable_diversity:
        print("提示: --no_enable_diversity 已忽略，当前版本固定启用 CoT 多样性。")
    args.enable_diversity = True

    print(f"\n配置:")
    print(f"  游戏数量: {args.num_games}")
    print(f"  难度: {args.difficulty}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  包含CoT: {args.with_thinking}")
    print("  CoT多样性: True (fixed)")
    print(f"  Expert搜索深度: {args.expert_depth}")
    print(f"  Expert空位分支上限: {args.expert_max_empty}")
    print(f"  最低最终分数: {args.min_final_score}")
    print(f"  最低步数: {args.min_steps}")
    print(f"  最小唯一状态比例: {args.min_unique_state_ratio}")
    print()

    if args.difficulty == 'mixed':
        games = generate_mixed_data(
            args.num_games,
            args.seed,
            args.with_thinking,
            args.enable_diversity,
            args.expert_depth,
            args.expert_max_empty,
        )
    else:
        games = generate_games(
            num_games=args.num_games,
            difficulty=args.difficulty,
            seed=args.seed,
            with_thinking=args.with_thinking,
            enable_diversity=args.enable_diversity,
            expert_depth=args.expert_depth,
            expert_max_empty=args.expert_max_empty,
        )

    # Lightweight quality filtering
    before_count = len(games)
    games = filter_games_by_quality(
        games,
        min_final_score=args.min_final_score,
        min_steps=args.min_steps,
        min_unique_state_ratio=args.min_unique_state_ratio,
    )
    after_count = len(games)
    if after_count != before_count:
        print(f"质量过滤: {before_count} -> {after_count} games")

    save_games(games, args.output_dir)

    # Quality report
    report = build_data_quality_report(games)
    print("\n=== Data Quality Report ===")
    print(f"Total games: {report['num_games']}")
    print(f"Total steps: {report['num_steps']}")
    print(f"Mean final score: {report['mean_final_score']:.2f}")
    print(f"Max final score: {report['max_final_score']}")
    print(f"Min final score: {report['min_final_score']}")
    print(f"Mean max tile: {report['mean_max_tile']:.2f}")
    print(f"Max tile reached: {report['max_tile_reached']}")
    print(f"2048 reached: {report['tile_2048_games']} games")
    print(f"Overall unique state ratio: {report['overall_unique_state_ratio']:.3f}")
    print(f"Mean unique state ratio/game: {report['mean_unique_state_ratio_per_game']:.3f}")
    print(f"CoT step coverage: {report['cot_step_coverage']:.3f}")
    print(f"Mean thinking length(chars): {report['mean_thinking_length_chars']:.1f}")
    print(f"High-info CoT ratio: {report['high_info_cot_ratio']:.3f}")
    print(f"Difficulty distribution: {report['difficulty_distribution']}")
    print(f"Action distribution: {report['action_distribution']}")

    if args.report_file:
        report_path = Path(args.report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(_convert_numpy_types(report), f, ensure_ascii=False, indent=2)
        print(f"质量报告已保存: {report_path}")

    manifest = build_generation_manifest(
        args=args,
        report=report,
        output_dir=args.output_dir,
    )
    manifest_path = Path(args.manifest_file) if args.manifest_file else Path(args.output_dir) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(_convert_numpy_types(manifest), f, ensure_ascii=False, indent=2)
    print(f"数据manifest已保存: {manifest_path}")


if __name__ == "__main__":
    main()
