"""Box backflip MDP terms.

The duck starts STANDING on the edge of a box (boxflip_terrain.py) and has to
throw itself backward off the edge, rotate a full turn in the air and land on
its feet on the crash mat below. Nothing is scripted and nothing spawns
mid-air: every episode begins on the box top at the home pose.

Why not reuse the roulade terms directly: the roulade accumulator is
SUPPORT-GATED (rotation only counts while touching the ground) and its landing
annuity requires a head-plant latch — both exactly wrong for a ballistic flip.
This module keeps the roulade's proven shape (one potential-based progress
signal, state-gated landing rewards) with those two gates removed, and adds
the penalties a flip needs: body-on-terrain (no rolling along the box top,
no crash landings) and over-rotation.

Sign convention: forward roll = +ω_y (body frame), so a BACKflip is −ω_y.
The accumulator stores the backward rotation as a positive number.

Per-env state on the env object (created lazily, reset by reset_boxflip_state):
  env._flip_accum, env._flip_max, env._flip_paid
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks.mdp import (
    _lateral_axis_z,
    _sensor_any_contact,
    _servo_default_joint_pos,
    _servo_joint_ids,
    _servo_joint_pos,
)

_ASSET = SceneEntityCfg("robot")
FEET_SENSOR = "feet_ground_contact"
ROBOT_SENSOR = "robot_ground_contact"
HEAD_SENSOR = "head_ground_contact"

# Landing pose: every servo near HOME. Std in rad (0.4 ≈ 23°, the roulade's
# value). The 2026-09-09 boxflip-a100 run found the head-tripod exploit —
# trunk upright at standing height with the head planted on the mat behind
# and the legs folded in front — which satisfies feet+upright+height exactly.
# A pose Gaussian over ALL 14 servos (neck/head included) and a head-free
# factor close that hole.
_POSE_STD = 0.4

# Flatness gate on the accumulator (borrowed from the roulade): a flip is a
# pure pitch; rotation while tipped over onto a shoulder earns nothing.
_FLAT_FULL, _FLAT_ZERO = 0.5, 0.866


def _state(env: ManagerBasedRlEnv):
    if not hasattr(env, "_flip_accum"):
        z = torch.zeros(env.num_envs, device=env.device)
        env._flip_accum = z.clone()
        env._flip_max = z.clone()
        env._flip_paid = z.clone()
        env._flip_dirty = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._flip_dirty_steps = z.clone()
        env._flip_touched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._flip_first_contact_bonus = z.clone()
        env._flip_last_step = -1
    return env._flip_accum, env._flip_max, env._flip_paid


def _update(env: ManagerBasedRlEnv, asset: Entity) -> None:
    """Integrate backward pitch rate, on the box, in the air and on the mat alike."""
    _state(env)
    step = int(env.common_step_counter)
    if step == env._flip_last_step:
        return
    omega_back = -asset.data.root_link_ang_vel_b[:, 1]
    delta = torch.nan_to_num(omega_back, nan=0.0) * env.step_dt
    y_z = torch.nan_to_num(_lateral_axis_z(asset.data.root_link_quat_w), nan=1.0).abs()
    t = torch.clamp((_FLAT_ZERO - y_z) / (_FLAT_ZERO - _FLAT_FULL), 0.0, 1.0)
    delta = delta * (t * t * (3.0 - 2.0 * t))
    env._flip_accum = env._flip_accum + delta
    env._flip_max = torch.maximum(env._flip_max, env._flip_accum)
    # Clean-landing latch (user rule, 2026-09-10): the head must NEVER touch
    # anything, and on the mat nothing but the feet may touch. One violation
    # marks the episode dirty for good and every landing reward pays zero.
    head_hit = (1.0 - _head_free(env)).bool()
    below_box = asset.data.root_link_pos_w[:, 2] < env.scene.env_origins[:, 2] - 0.05
    body_hit = _body_on_terrain(env).bool() & below_box
    dirty_now = head_hit | body_hit
    env._flip_dirty = env._flip_dirty | dirty_now
    env._flip_dirty_steps = env._flip_dirty_steps + dirty_now.float()
    # First terrain contact below the box top: record a one-time bonus for
    # how much of the turn was done before touching down, feet-first only.
    # This is the dense slope from "head tap at ~290°" toward "feet at 360°"
    # that the binary latch lacked (v3 froze at landing=0 for 400 iters).
    rc = _sensor_any_contact(env, ROBOT_SENSOR)
    rc = rc if rc is not None else torch.zeros_like(below_box)
    first = rc & below_box & ~env._flip_touched
    deficit = torch.clamp(2 * math.pi - env._flip_max, min=0.0)
    # No feet factor here (v2's first contact is always the head → a feet
    # gate would be zero everywhere, no slope). Rotation alone, wide std:
    # 290° → 0.03, 320° → 0.37, 340° → 0.78, 360° → 1. Head/body contact is
    # priced by the penalties and the soft clean factor instead.
    quality = torch.exp(-(deficit / math.radians(40.0)) ** 2)
    env._flip_first_contact_bonus = torch.where(first, quality, torch.zeros_like(quality))
    env._flip_touched = env._flip_touched | first
    env._flip_last_step = step


def _gate(env: ManagerBasedRlEnv, lo: float, hi: float) -> torch.Tensor:
    _, mx, _ = _state(env)
    t = torch.clamp((mx - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _upright(asset: Entity) -> torch.Tensor:
    """1 when gravity points straight down the trunk's -z, 0 at 90° or beyond."""
    g = asset.data.projected_gravity_b
    return torch.clamp(-torch.nan_to_num(g[:, 2], nan=0.0), 0.0, 1.0) ** 2


def _feet(env: ManagerBasedRlEnv) -> torch.Tensor:
    c = _sensor_any_contact(env, FEET_SENSOR)
    return c.float() if c is not None else torch.zeros(env.num_envs, device=env.device)


def _body_on_terrain(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Robot touching the terrain with something other than the feet.

    Approximation: whole-robot contact while the feet report none. A foot and
    a knee on the mat together read as 'feet', which is the lenient direction.
    """
    robot = _sensor_any_contact(env, ROBOT_SENSOR)
    if robot is None:
        return torch.zeros(env.num_envs, device=env.device)
    feet = _sensor_any_contact(env, FEET_SENSOR)
    feet = feet if feet is not None else torch.zeros_like(robot)
    return (robot & ~feet).float()


def _head_free(env: ManagerBasedRlEnv) -> torch.Tensor:
    """1 while the head is NOT touching the terrain, 0 while it is."""
    c = _sensor_any_contact(env, HEAD_SENSOR)
    return torch.ones(env.num_envs, device=env.device) if c is None else (~c).float()


def _pose(env: ManagerBasedRlEnv, asset: Entity, std: float = _POSE_STD) -> torch.Tensor:
    """exp(-mean((q - HOME)²)/std²) over all servo joints."""
    q = _servo_joint_pos(env, asset)
    home = _servo_default_joint_pos(env, asset)
    err = torch.nan_to_num(q - home, nan=0.0).pow(2).mean(dim=-1)
    return torch.exp(-err / (std * std))


def _head_over_feet(env: ManagerBasedRlEnv, asset: Entity, std: float = 0.03) -> torch.Tensor:
    """exp(-(|head_xy - feet_mid_xy| / std)²): 1 when the head is directly
    above the feet (HOME offset is 0.6 cm), ~0.3 at 3 cm, ~0 at 6 cm."""
    if not hasattr(env, "_flip_head_ids"):
        env._flip_head_ids = (
            asset.body_names.index("jaw_soft"),
            asset.site_names.index("left_foot"),
            asset.site_names.index("right_foot"),
        )
    h, lf, rf = env._flip_head_ids
    feet = 0.5 * (asset.data.site_pos_w[:, lf, :2] + asset.data.site_pos_w[:, rf, :2])
    d2 = torch.nan_to_num(asset.data.body_link_pos_w[:, h, :2] - feet, nan=1.0).pow(2).sum(-1)
    return torch.exp(-d2 / (std * std))


def _clean(env: ManagerBasedRlEnv, tau_steps: float = 1.0) -> torch.Tensor:
    """exp(-dirty_steps / tau): 1 with no head contact and no non-foot mat
    contact so far, 0.37 after one dirty step, 0.14 after two, ~0 after five.
    Only a perfectly clean landing pays in full, but fewer dirty steps always
    pays more — the gradient a binary latch does not have."""
    _state(env)
    return torch.exp(-env._flip_dirty_steps / tau_steps)


def _landed(env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
    """feet on terrain × trunk upright × HOME pose × head off the ground
    × head directly over the feet × clean landing so far."""
    return (
        _feet(env) * _upright(asset) * _pose(env, asset) * _head_free(env)
        * _head_over_feet(env, asset) * _clean(env)
    )


# ── reset ─────────────────────────────────────────────────────────────────────

def reset_boxflip_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _ASSET,
    stand_z: float = 0.115,
    x_range: tuple = (-0.01, 0.01),
    yaw_range: tuple = (-0.05, 0.05),
    joint_noise_std: float = 0.03,
):
    """Standing on the box edge, home pose, facing +x (the edge is behind).

    The terrain origin is the spawn point on the box top (boxflip_terrain.py);
    mjlab has already put ``env.scene.env_origins`` there. The move is thrown
    BACKWARD, i.e. toward -x, off the edge and onto the mat.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    n = len(env_ids)
    asset: Entity = env.scene[asset_cfg.name]
    accum, mx, paid = _state(env)
    origins = env.scene.env_origins[env_ids]

    x = origins[:, 0] + torch.empty(n, device=env.device).uniform_(*x_range)
    y = origins[:, 1]
    z = origins[:, 2] + stand_z
    yaw = torch.empty(n, device=env.device).uniform_(*yaw_range)
    quat = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], dim=1)

    env.sim.data.qpos[env_ids, 0] = x
    env.sim.data.qpos[env_ids, 1] = y
    env.sim.data.qpos[env_ids, 2] = z
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qvel[env_ids, :6] = 0.0
    if joint_noise_std > 0.0:
        servo_ids = _servo_joint_ids(env, asset)
        cols = torch.tensor([7 + j for j in servo_ids], device=env.device, dtype=torch.long)
        env.sim.data.qpos[env_ids.unsqueeze(1), cols.unsqueeze(0)] += (
            torch.randn(n, len(cols), device=env.device) * joint_noise_std
        )
    accum[env_ids] = 0.0
    mx[env_ids] = 0.0
    paid[env_ids] = 0.0
    env._flip_dirty[env_ids] = False
    env._flip_dirty_steps[env_ids] = 0.0
    env._flip_touched[env_ids] = False
    env._flip_first_contact_bonus[env_ids] = 0.0


# ── rewards ───────────────────────────────────────────────────────────────────

def flip_progress(
    env: ManagerBasedRlEnv,
    target_angle: float = 2 * math.pi,
    asset_cfg: SceneEntityCfg = _ASSET,
) -> torch.Tensor:
    """Paid increments of the max-so-far backward rotation, up to one turn.

    Potential-based: a full flip pays exactly ``target_angle`` in total no
    matter how; standing still pays nothing; un-rotating pays nothing back.
    No rate cap — unlike the roulade, speed is the point here.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update(env, asset)
    _, mx, paid = _state(env)
    frontier = torch.clamp(mx, max=target_angle)
    inc = torch.clamp(frontier - paid, min=0.0)
    env._flip_paid = torch.maximum(paid, frontier)
    return inc


def flip_landing(
    env: ManagerBasedRlEnv,
    gate_lo: float = math.radians(300.0),
    gate_hi: float = math.radians(340.0),
    asset_cfg: SceneEntityCfg = _ASSET,
) -> torch.Tensor:
    """After (nearly) a full turn: feet on the terrain, trunk upright, joints
    at HOME, head off the ground.

    Product, so a face-plant after a full turn pays nothing, neither does an
    upright duck that never rotated, and neither does the head-tripod.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update(env, asset)
    return _gate(env, gate_lo, gate_hi) * _landed(env, asset)


def flip_height(
    env: ManagerBasedRlEnv,
    mat_top: float = 0.12,
    stand_z: float = 0.115,
    gate_lo: float = math.radians(300.0),
    gate_hi: float = math.radians(340.0),
    asset_cfg: SceneEntityCfg = _ASSET,
) -> torch.Tensor:
    """After the turn: trunk at standing height above the mat, upright.

    Heights are relative to the terrain origin (the box top) minus the box
    height: the mat top sits ``mat_top`` above the tile floor. Uses the env
    origin so it is right whatever the tile layout.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update(env, asset)
    box_top = env.scene.env_origins[:, 2]
    trunk = asset.data.root_link_pos_w[:, 2]
    floor = box_top - _box_height(env)
    h = torch.clamp((trunk - floor - mat_top) / stand_z, 0.0, 1.0)
    return _gate(env, gate_lo, gate_hi) * h * _upright(asset) * _pose(env, asset) * _head_free(env) * _clean(env)


def flip_still(
    env: ManagerBasedRlEnv,
    gate_lo: float = math.radians(300.0),
    gate_hi: float = math.radians(340.0),
    asset_cfg: SceneEntityCfg = _ASSET,
) -> torch.Tensor:
    """After the turn, upright and on the feet: reward coming to rest."""
    asset: Entity = env.scene[asset_cfg.name]
    _update(env, asset)
    v = torch.nan_to_num(asset.data.root_link_lin_vel_b, nan=0.0).pow(2).sum(-1)
    w = torch.nan_to_num(asset.data.root_link_ang_vel_b, nan=0.0).pow(2).sum(-1)
    return _gate(env, gate_lo, gate_hi) * _landed(env, asset) * torch.exp(-(v + 0.1 * w))


def flip_body_on_terrain(
    env: ManagerBasedRlEnv,
    above_box_only: bool = True,
    asset_cfg: SceneEntityCfg = _ASSET,
) -> torch.Tensor:
    """1 while the body (not the feet) touches the terrain. Negative weight.

    Run-1 lesson (2026-09-09, iter 250): charged everywhere, this made a
    failed attempt (fall off, lie on the mat for 2 s) cost ~1.0 against ~0.3
    of rotation credit, and the policy converged to standing still on the
    box. With ``above_box_only`` it is charged only while the trunk is above
    the box top — its real job, stopping the roulade-style roll along the box
    — and a crash on the mat is left to the landing rewards (which simply pay
    nothing) so that trying is never worse than not trying.
    """
    hit = _body_on_terrain(env)
    if above_box_only:
        asset: Entity = env.scene[asset_cfg.name]
        box_top = env.scene.env_origins[:, 2]
        hit = hit * (asset.data.root_link_pos_w[:, 2] > box_top - 0.05).float()
    return hit


def flip_first_contact(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ASSET) -> torch.Tensor:
    """One-time bonus on the step of first terrain contact below the box:
    exp(-((360° - rotation)/40°)²) — pays for finishing the turn in the air."""
    asset: Entity = env.scene[asset_cfg.name]
    _update(env, asset)
    return env._flip_first_contact_bonus


def flip_head_on_terrain(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ASSET) -> torch.Tensor:
    """1 while the head touches the box or the mat. Negative weight.

    Direct tax on the head-tripod landing (and on head-first crashes). The
    flip is ballistic, so there is no legitimate head contact at any point.
    """
    return 1.0 - _head_free(env)


def flip_overrotation(
    env: ManagerBasedRlEnv,
    target_angle: float = 2 * math.pi,
    slack: float = math.radians(25.0),
    asset_cfg: SceneEntityCfg = _ASSET,
) -> torch.Tensor:
    """(rotation beyond one turn + slack)²: tumbling on after landing costs."""
    asset: Entity = env.scene[asset_cfg.name]
    _update(env, asset)
    accum, _, _ = _state(env)
    return torch.clamp(accum - target_angle - slack, min=0.0).pow(2)


def flip_sagittal(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ASSET) -> torch.Tensor:
    """ω_x² + ω_z²: rotation out of the flip plane. Negative weight."""
    asset: Entity = env.scene[asset_cfg.name]
    w = torch.nan_to_num(asset.data.root_link_ang_vel_b, nan=0.0)
    return w[:, 0].pow(2) + w[:, 2].pow(2)


def _box_height(env: ManagerBasedRlEnv) -> torch.Tensor:
    if not hasattr(env, "_flip_box_height"):
        from mjlab_microduck.tasks.boxflip_terrain import BOX_HEIGHT
        env._flip_box_height = torch.full((env.num_envs,), BOX_HEIGHT, device=env.device)
    return env._flip_box_height


# ── metrics ───────────────────────────────────────────────────────────────────

def flip_max_rotation_deg(env: ManagerBasedRlEnv) -> torch.Tensor:
    _, mx, _ = _state(env)
    return torch.rad2deg(mx)
