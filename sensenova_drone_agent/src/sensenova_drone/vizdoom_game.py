"""ViZDoom adapter: second visual domain for the thinking-in-frames stack.

Mirrors the DroneMazeEnv interface exactly — reset(seed)/step(a)->5-tuple/
render()->HWC-RGB/expert_action_index()/snapshot()/restore() and the
terminal-reward conventions (success > +5, fatal < -4) — so every
collector, DAgger script, and closed-loop evaluator works by swapping the
env class. Scenario: deadly_corridor (reach the armor at the corridor end;
enemies en route; death = failure).

Action indices keep the drone game's 9-slot semantic layout:
  0 noop, 1 turn_left, 2 turn_right, 3 attack (was ascend), 4 noop
  (was descend), 5 forward, 6 backward, 7 strafe_left, 8 strafe_right.

Privileged-state oracle (the teacher): attack when a live enemy is near
the crosshair, else steer toward the armor and advance. Snapshot/restore
uses ViZDoom save-states (temp files) — enables expert chunk labels and
counterfactual branches exactly as in the drone game.
"""
from __future__ import annotations

import math
import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import vizdoom as vzd

NUM_ACTIONS = 9
ACTION_LABELS = ["noop", "turn_left", "turn_right", "attack", "noop2",
                 "forward", "backward", "strafe_left", "strafe_right"]
# 9 semantic slots -> per-scenario button vectors.
# deadly_corridor buttons: MOVE_LEFT MOVE_RIGHT ATTACK MOVE_FORWARD MOVE_BACKWARD TURN_LEFT TURN_RIGHT
_CORRIDOR_BUTTONS = {
    0: [0, 0, 0, 0, 0, 0, 0], 1: [0, 0, 0, 0, 0, 1, 0], 2: [0, 0, 0, 0, 0, 0, 1],
    3: [0, 0, 1, 0, 0, 0, 0], 4: [0, 0, 0, 0, 0, 0, 0], 5: [0, 0, 0, 1, 0, 0, 0],
    6: [0, 0, 0, 0, 1, 0, 0], 7: [1, 0, 0, 0, 0, 0, 0], 8: [0, 1, 0, 0, 0, 0, 0],
}
# health_gathering buttons: TURN_LEFT TURN_RIGHT MOVE_FORWARD
_HEALTH_BUTTONS = {
    0: [0, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 0], 4: [0, 0, 0],
    5: [0, 0, 1], 6: [0, 0, 0], 7: [0, 0, 0], 8: [0, 0, 0],
}
_SCENARIOS = {
    # cfg name -> (buttons, goal object names, survive_to_win)
    "health_gathering": (_HEALTH_BUTTONS, {"CustomMedikit", "Medikit"}, True),
    "deadly_corridor": (_CORRIDOR_BUTTONS, {"GreenArmor", "Armor"}, False),
}
_ENEMY_NAMES = {"Zombieman", "ShotgunGuy", "ChaingunGuy", "DoomImp", "Demon"}


@dataclass
class VizdoomGameConfig:
    scenario: str = "health_gathering"
    max_episode_steps: int = 160
    frame_skip: int = 4
    skill: int = 3            # 5 = scenario default, brutal; 3 keeps a heuristic teacher viable
    reward_scale: float = 0.01  # scenario's built-in dX shaping is ~O(100)/episode
    success_reward: float = 10.0
    death_reward: float = -5.0
    attack_cone_deg: float = 8.0
    turn_deadband_deg: float = 15.0


class VizdoomCorridorEnv:
    def __init__(self, cfg: VizdoomGameConfig | None = None):
        self.cfg = cfg or VizdoomGameConfig()
        g = vzd.DoomGame()
        self._buttons, self._goal_names, self._survive_to_win = _SCENARIOS[self.cfg.scenario]
        g.load_config(os.path.join(vzd.scenarios_path, f"{self.cfg.scenario}.cfg"))
        g.set_doom_skill(int(self.cfg.skill))
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_window_visible(False)
        g.set_objects_info_enabled(True)
        for v in (vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y,
                  vzd.GameVariable.ANGLE, vzd.GameVariable.HEALTH,
                  vzd.GameVariable.KILLCOUNT):
            g.add_available_game_variable(v)
        g.init()
        self.game = g
        self._step_index = 0
        self._last_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self._goal_xy: tuple[float, float] | None = None
        self._tmpdir = tempfile.mkdtemp(prefix="vzd_snap_")
        self._attack_streak = 0
        self._oracle_cooldown = 0
        self._last_kills = 0.0

    # ------------------------------------------------------------------ core
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.game.set_seed(int(seed) % (2 ** 31 - 1))
        self.game.new_episode()
        self._step_index = 0
        self._attack_streak = 0
        self._oracle_cooldown = 0
        self._last_kills = 0.0
        self._goal_xy = None
        self._refresh(cache_goal=True)
        return self._last_frame, {}

    def step(self, action: int):
        r_raw = self.game.make_action(self._buttons[int(action)], int(self.cfg.frame_skip))
        self._step_index += 1
        self._refresh()
        dead = self.game.is_player_dead()
        finished = self.game.is_episode_finished()
        if self._survive_to_win:
            success = (not dead) and (self._step_index >= self.cfg.max_episode_steps or finished)
        else:
            success = finished and not dead and self._near_goal()
        terminated = dead or success
        truncated = False
        if success:
            reward = self.cfg.success_reward
        elif dead:
            reward = self.cfg.death_reward
        else:
            reward = float(r_raw) * self.cfg.reward_scale
        return self._last_frame, reward, terminated, truncated, {}

    def render(self) -> np.ndarray:
        return self._last_frame

    # ---------------------------------------------------------------- oracle
    def expert_action_index(self) -> int:
        st = self.game.get_state()
        if st is None:
            return 0
        # corpses keep their actor names in the objects list, so raw
        # name-matching attacks forever; gate combat on kill PROGRESS —
        # an attack streak that doesn't raise KILLCOUNT means the matched
        # "enemy" is a corpse (or unreachable): cool down and navigate.
        kills = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        if kills > self._last_kills:
            self._attack_streak = 0
        self._last_kills = kills
        if self._oracle_cooldown > 0:
            self._oracle_cooldown -= 1
        px = self.game.get_game_variable(vzd.GameVariable.POSITION_X)
        py = self.game.get_game_variable(vzd.GameVariable.POSITION_Y)
        ang = self.game.get_game_variable(vzd.GameVariable.ANGLE)
        if self._oracle_cooldown == 0:
            best = None
            for o in st.objects or []:
                if o.name in _ENEMY_NAMES:
                    rel = self._rel_angle(px, py, ang, o.position_x, o.position_y)
                    d = math.hypot(o.position_x - px, o.position_y - py)
                    if abs(rel) <= self.cfg.attack_cone_deg and (best is None or d < best):
                        best = d
            if best is not None:
                self._attack_streak += 1
                if self._attack_streak > 8:
                    self._attack_streak = 0
                    self._oracle_cooldown = 12
                else:
                    return 3
        self._attack_streak = 0
        # steer to the goal and advance
        if self._goal_xy is not None:
            rel = self._rel_angle(px, py, ang, *self._goal_xy)
            if rel > self.cfg.turn_deadband_deg:
                return 1
            if rel < -self.cfg.turn_deadband_deg:
                return 2
        return 5

    # ------------------------------------------------------- snapshot/restore
    def snapshot(self) -> dict[str, Any]:
        path = os.path.join(self._tmpdir, f"{uuid.uuid4().hex}.save")
        self.game.save(path)
        return {"path": path, "step_index": self._step_index, "goal": self._goal_xy}

    def restore(self, snapshot: dict[str, Any]) -> None:
        # ZDoom applies load() on the next tic; a finished/dead episode never
        # tics, so the load would silently never happen. new_episode() first.
        if self.game.is_episode_finished() or self.game.is_player_dead():
            self.game.new_episode()
        self.game.load(snapshot["path"])
        self._step_index = snapshot["step_index"]
        self._goal_xy = snapshot["goal"]
        self._refresh()

    # -------------------------------------------------------------- internals
    def _refresh(self, cache_goal: bool = False):
        st = self.game.get_state()
        if st is not None and st.screen_buffer is not None:
            self._last_frame = np.ascontiguousarray(st.screen_buffer)
        if st is not None and (cache_goal or self._goal_xy is None or self._survive_to_win):
            px = self.game.get_game_variable(vzd.GameVariable.POSITION_X)
            py = self.game.get_game_variable(vzd.GameVariable.POSITION_Y)
            best = None
            for o in st.objects or []:
                if o.name in self._goal_names:
                    d = math.hypot(o.position_x - px, o.position_y - py)
                    if best is None or d < best[0]:
                        best = (d, (o.position_x, o.position_y))
            if best is not None:
                self._goal_xy = best[1]

    def _near_goal(self) -> bool:
        if self._goal_xy is None:
            return True  # scenario ended alive without goal info: count pickup-end as success
        px = self.game.get_game_variable(vzd.GameVariable.POSITION_X)
        py = self.game.get_game_variable(vzd.GameVariable.POSITION_Y)
        return math.hypot(self._goal_xy[0] - px, self._goal_xy[1] - py) < 120.0

    @staticmethod
    def _rel_angle(px, py, ang_deg, tx, ty) -> float:
        target = math.degrees(math.atan2(ty - py, tx - px))
        rel = (target - ang_deg + 180.0) % 360.0 - 180.0
        return rel

    def close(self):
        self.game.close()
