"""
2048 Game Environment
Implementation of the 2048 game logic for training LLMs.
"""

import numpy as np
import random
import re
import json
from typing import Tuple, List, Optional


# Action mappings
ACTION_MAP = {
    0: "上",
    1: "右",
    2: "下",
    3: "左"
}

ACTION_MAP_ENGLISH = {
    0: "UP",
    1: "RIGHT",
    2: "DOWN",
    3: "LEFT",
}

ACTION_NAMES_ENG = {
    "up": 0,
    "right": 1,
    "down": 2,
    "left": 3
}

ACTION_NAMES_CHI = {
    "上": 0,
    "右": 1,
    "下": 2,
    "左": 3
}

_ACTION_TOKEN_TO_ID = {
    "上": 0,
    "右": 1,
    "下": 2,
    "左": 3,
    "up": 0,
    "right": 1,
    "down": 2,
    "left": 3,
}

_TRAILING_TEMPLATE_TOKEN_RE = re.compile(
    r"(?:\s*(?:<\|[^>\n]+\|>|</s>|<\s*/s\s*>))+\s*$"
)
_LEADING_TEMPLATE_TOKEN_RE = re.compile(
    r"^(?:\s*(?:<\|[^>\n]+\|>|</s>|<\s*/s\s*>))+"
)
_THINK_CLOSE_RE = re.compile(r"</think>", flags=re.I)
_EN_TOP_LEVEL_KEYS = {"board", "judgment", "choice"}
_EN_SITUATION_KEYS = {"max_tile", "positions", "in_corner"}
_EN_JUDGMENT_KEYS = {"UP", "RIGHT", "DOWN", "LEFT"}
_ZH_TOP_LEVEL_KEYS = {"局面", "判断", "选择"}
_ZH_SITUATION_KEYS = {"最大数字", "位置", "在角落"}
_ZH_JUDGMENT_KEYS = {"上", "右", "下", "左"}


def _normalize_action_text(text: str) -> str:
    normalized = (text or "").rstrip()
    while True:
        stripped = _TRAILING_TEMPLATE_TOKEN_RE.sub("", normalized)
        if stripped == normalized:
            break
        normalized = stripped.rstrip()
    while True:
        stripped = _LEADING_TEMPLATE_TOKEN_RE.sub("", normalized)
        if stripped == normalized:
            break
        normalized = stripped.lstrip()
    return normalized


def _decode_action_token(token: str) -> Optional[int]:
    token_norm = str(token).strip().lower()
    if not token_norm:
        return None
    return _ACTION_TOKEN_TO_ID.get(token_norm)


def _extract_non_think_output(text: str) -> str:
    normalized = _normalize_action_text(text)
    if not normalized:
        return ""

    closes = list(_THINK_CLOSE_RE.finditer(normalized))
    if closes:
        return normalized[closes[-1].end():].strip()

    # No think block present: treat whole output as non-think region.
    return normalized.strip()


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_valid_position_list(value: object) -> bool:
    if not isinstance(value, list):
        return False
    for coord in value:
        if not isinstance(coord, list) or len(coord) != 2:
            return False
        if not all(_is_plain_int(v) for v in coord):
            return False
        if not (0 <= coord[0] <= 3 and 0 <= coord[1] <= 3):
            return False
    return True


def _validate_action_json_schema(obj: object) -> Optional[int]:
    if not isinstance(obj, dict):
        return None

    if set(obj.keys()) == _EN_TOP_LEVEL_KEYS:
        situation = obj.get("board")
        judgment = obj.get("judgment")
        choice = obj.get("choice")

        if not isinstance(situation, dict) or set(situation.keys()) != _EN_SITUATION_KEYS:
            return None
        if not _is_plain_int(situation.get("max_tile")):
            return None
        if not _is_valid_position_list(situation.get("positions")):
            return None
        if not isinstance(situation.get("in_corner"), bool):
            return None

        if not isinstance(judgment, dict) or set(judgment.keys()) != _EN_JUDGMENT_KEYS:
            return None
        if not all(isinstance(v, bool) for v in judgment.values()):
            return None

        if not isinstance(choice, str):
            return None
        choice_norm = choice.strip().upper()
        action_id = _decode_action_token(choice_norm)
        if action_id is None:
            return None
        if judgment.get(choice_norm) is not True:
            return None
        return int(action_id)

    if set(obj.keys()) == _ZH_TOP_LEVEL_KEYS:
        situation = obj.get("局面")
        judgment = obj.get("判断")
        choice = obj.get("选择")

        if not isinstance(situation, dict) or set(situation.keys()) != _ZH_SITUATION_KEYS:
            return None
        if not _is_plain_int(situation.get("最大数字")):
            return None
        if not _is_valid_position_list(situation.get("位置")):
            return None
        if not isinstance(situation.get("在角落"), bool):
            return None

        if not isinstance(judgment, dict) or set(judgment.keys()) != _ZH_JUDGMENT_KEYS:
            return None
        if not all(isinstance(v, bool) for v in judgment.values()):
            return None

        if not isinstance(choice, str):
            return None
        choice_norm = choice.strip()
        action_id = _decode_action_token(choice_norm)
        if action_id is None:
            return None
        if judgment.get(choice_norm) is not True:
            return None
        return int(action_id)

    return None


def parse_action_from_non_think_text(text: str) -> Optional[int]:
    """Parse action from non-think output JSON only.

    Strict mode:
    - Only parse non-think region (text after the last `</think>` when present).
    - Non-think region must be valid JSON and pass action schema checks.
    - Returns None when JSON is invalid or schema/choice is invalid.
    """
    candidate = _extract_non_think_output(text)
    if not candidate:
        return None

    try:
        parsed = json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

    return _validate_action_json_schema(parsed)


class Game2048:
    """2048 game environment"""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the game.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.grid = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.game_over = False
        self.reset()

    def reset(self, initial_tiles: int = 2, prob_4: float = 0.1) -> str:
        """Reset the game and return initial state."""
        self.grid = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.game_over = False
        initial_tiles = int(np.clip(int(initial_tiles), 1, 16))
        prob_4 = float(np.clip(float(prob_4), 0.0, 1.0))
        for _ in range(initial_tiles):
            self._add_new_tile(prob_4=prob_4)
        return self._get_state()

    def _add_new_tile(self, prob_4: float = 0.1) -> None:
        """Add a new tile (2 or 4) to a random empty cell."""
        empty_cells = [(i, j) for i in range(4) for j in range(4) if self.grid[i, j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.grid[i, j] = 4 if random.random() < prob_4 else 2

    def step(self, action: int) -> Tuple[str, float, bool, int]:
        """
        Execute an action.

        Args:
            action: 0=up, 1=right, 2=down, 3=left

        Returns:
            (next_state, reward, done, score)
        """
        prev_score = self.score
        moved = self._move(action)

        if moved:
            self._add_new_tile()
            reward = self.score - prev_score
        else:
            # Penalty for invalid move
            reward = -1

        done = self._is_game_over()
        state = self._get_state()

        return state, reward, done, self.score

    def _move(self, action: int) -> bool:
        """
        Execute a move and return whether the board changed.

        Args:
            action: 0=up, 1=right, 2=down, 3=left

        Returns:
            True if the board changed, False otherwise
        """
        if action == 0:  # Up
            return self._move_up()
        elif action == 1:  # Right
            return self._move_right()
        elif action == 2:  # Down
            return self._move_down()
        elif action == 3:  # Left
            return self._move_left()
        return False

    def _move_left(self) -> bool:
        """Move and merge tiles to the left"""
        moved = False
        for i in range(4):
            row = self.grid[i, :]
            new_row, row_moved, row_score = self._compress_and_merge(row)
            if row_moved:
                moved = True
                self.grid[i, :] = new_row
                self.score += row_score
        return moved

    def _move_right(self) -> bool:
        """Move and merge tiles to the right"""
        moved = False
        for i in range(4):
            row = self.grid[i, :][::-1]  # Reverse
            new_row, row_moved, row_score = self._compress_and_merge(row)
            if row_moved:
                moved = True
                self.grid[i, :] = new_row[::-1]
                self.score += row_score
        return moved

    def _move_up(self) -> bool:
        """Move and merge tiles up"""
        moved = False
        for j in range(4):
            col = self.grid[:, j]
            new_col, col_moved, col_score = self._compress_and_merge(col)
            if col_moved:
                moved = True
                self.grid[:, j] = new_col
                self.score += col_score
        return moved

    def _move_down(self) -> bool:
        """Move and merge tiles down"""
        moved = False
        for j in range(4):
            col = self.grid[:, j][::-1]  # Reverse
            new_col, col_moved, col_score = self._compress_and_merge(col)
            if col_moved:
                moved = True
                self.grid[:, j] = new_col[::-1]
                self.score += col_score
        return moved

    def _compress_and_merge(self, line: np.ndarray) -> Tuple[np.ndarray, bool, int]:
        """
        Compress and merge a line (row or column).

        Args:
            line: A 1D array of 4 elements

        Returns:
            (new_line, moved, score_gain)
        """
        # Remove zeros
        non_zero = line[line > 0]
        original = line.copy()

        # Merge adjacent equal tiles
        merged = []
        score = 0
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                # Merge
                merged_value = non_zero[i] * 2
                merged.append(merged_value)
                score += merged_value
                i += 2
            else:
                merged.append(non_zero[i])
                i += 1

        # Pad with zeros
        result = np.zeros(4, dtype=int)
        result[:len(merged)] = merged

        moved = not np.array_equal(original, result)
        return result, moved, score

    def _is_game_over(self) -> bool:
        """Check if the game is over"""
        # Check for empty cells
        if np.any(self.grid == 0):
            return False

        # Check for possible merges horizontally
        for i in range(4):
            for j in range(3):
                if self.grid[i, j] == self.grid[i, j + 1]:
                    return False

        # Check for possible merges vertically
        for i in range(3):
            for j in range(4):
                if self.grid[i, j] == self.grid[i + 1, j]:
                    return False

        return True

    def _get_state(self) -> str:
        """Get the current state as 4x4 array format (方便模型解读)"""
        return self._grid_to_array()

    def _grid_to_array(self) -> str:
        """
        Convert the grid to 4x4 array format.

        格式：
        [[2, 0, 4, 0],
         [0, 8, 0, 0],
         [0, 0, 0, 0],
         [16, 0, 0, 2]]
        """
        lines = []
        for row in self.grid:
            line = "[" + ", ".join(str(x) for x in row) + "]"
            lines.append(line)
        return "[\n" + ",\n".join(lines) + "\n]"

    def _grid_to_text(self) -> str:
        """Convert the grid to text format (deprecated, 保留兼容性)"""
        lines = []
        for row in self.grid:
            line = " ".join(f"{x:4d}" if x > 0 else "   ." for x in row)
            lines.append(line)
        return "\n".join(lines)

    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions"""
        actions = []
        for action in range(4):
            if self._can_move(action):
                actions.append(action)
        return actions

    def _can_move(self, action: int) -> bool:
        """
        Check if a move is valid (would change the board).

        Args:
            action: 0=up, 1=right, 2=down, 3=left

        Returns:
            True if the move would change the board
        """
        # Save current state (both grid and score)
        original_grid = self.grid.copy()
        original_score = self.score

        # Simulate the move
        self._move(action)

        # Check if grid changed
        changed = not np.array_equal(original_grid, self.grid)

        # Restore original state (both grid and score)
        self.grid = original_grid
        self.score = original_score

        return changed

    def get_max_tile(self) -> int:
        """Get the maximum tile value on the board"""
        return int(np.max(self.grid))

    def get_empty_cells(self) -> int:
        """Get the number of empty cells"""
        return int(np.sum(self.grid == 0))

    def clone(self) -> 'Game2048':
        """Create a deep copy of the game"""
        new_game = Game2048()
        new_game.grid = self.grid.copy()
        new_game.score = self.score
        new_game.game_over = self.game_over
        return new_game


def parse_action_from_text(text: str) -> int:
    """
    Parse action from model output text.

    Args:
        text: Model output text

    Returns:
        Action ID (0-3), or -1 when output is invalid/illegal.
    """
    non_think_action = parse_action_from_non_think_text(text)
    if non_think_action is not None:
        return int(non_think_action)

    return -1


if __name__ == "__main__":
    # Quick test
    game = Game2048()
    print("Initial state:")
    print(game._get_state())
    print(f"Valid actions: {game.get_valid_actions()}")
    print(f"Max tile: {game.get_max_tile()}")
