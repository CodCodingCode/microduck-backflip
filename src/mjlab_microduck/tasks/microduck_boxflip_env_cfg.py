"""Microduck box backflip task — Mjlab-BoxFlip-Flat-MicroDuck.

The duck spawns STANDING on the edge of a 0.8 m box (boxflip_terrain.py) at
the home pose and must throw itself backward off the edge, complete a full
turn in the air and land on its feet on the crash mat below. No mid-air
spawns, no scripted launch, no phase clock: the whole move is the policy's.

Why this is feasible where a floor backflip is not (measured 2026-09-09, see
the box-flip study): from flat ground the XL330 legs give ~60 ms of flight;
off a box edge the spin comes from tipping about the edge (gravity), which
reaches 13–15 rad/s, and the drop gives 250–350 ms of air. An open-loop kick
and tuck already completes 340–360° and lands feet-first on a 0.7–0.9 m box;
this task asks PPO to find that, and the landing, on its own.

Built on the roulade env (same DR, obs, actions, regularisers, symmetry) with
the roulade-specific pieces swapped out:
  • terrain: floor + box + soft mat instead of a plane;
  • reset: standing on the box edge only (reset_boxflip_state);
  • rewards: the flip set in boxflip_mdp.py — an UNGATED rotation accumulator
    (the roulade's support gate would zero out a ballistic flip), landing
    rewards gated on ≥300° of rotation, and penalties for body-on-terrain
    (rolling along the box top, crash landings) and over-rotation.
"""

from __future__ import annotations

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg, RewardTermCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab_microduck.tasks import boxflip_mdp
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.boxflip_terrain import (
    BOX_HEIGHT,
    BoxFlipTerrainCfg,
    MAT_THICKNESS,
)
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    MicroduckRouladeRlCfg,
    make_microduck_roulade_env_cfg,
)

TILE_SIZE = (5.0, 3.0)   # x: floor 0..5 with the box edge at x=2; y: 3 m wide
EPISODE_LENGTH_S = 3.5   # lean + flip ≈ 1 s, land and settle ≈ 2 s


def make_microduck_boxflip_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_roulade_env_cfg(play=play, direction=-1.0)
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Terrain: floor + box + mat, one tile, every env on the same box ──────
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=TILE_SIZE,
            curriculum=False,
            num_rows=1,
            num_cols=1,
            difficulty_range=(0.0, 0.0),
            sub_terrains={"box_flip": BoxFlipTerrainCfg()},
        ),
        max_init_terrain_level=0,
    )
    for k in ("terrain_levels",):
        cfg.curriculum.pop(k, None)

    # ── Reset: on the box edge, standing, that is all ─────────────────────────
    cfg.events.pop("set_roulade_state", None)
    cfg.events["set_boxflip_state"] = EventTermCfg(
        func=boxflip_mdp.reset_boxflip_state,
        mode="reset",
        params={
            "stand_z": 0.115,
            "x_range": (-0.01, 0.01),
            "yaw_range": (-0.05, 0.05),
            "joint_noise_std": 0.03,
        },
    )
    cfg.curriculum.pop("roulade_spawn_mix", None)

    # ── Rewards ───────────────────────────────────────────────────────────────
    for k in [k for k in cfg.rewards if k.startswith("roulade_")]:
        cfg.rewards.pop(k)
    # The flip is a large-angular-velocity event by definition; the generic
    # motion taxes would fight discovery (roulade/standup lesson).
    for k in ("body_ang_vel", "angular_momentum", "arrival_damping", "gentle_landing", "joint_torque_rate_l2"):
        if k in cfg.rewards:
            cfg.rewards[k].weight = 0.0
    for k in ("arrival_damping_weight", "torque_rate_weight", "gentle_landing_weight"):
        cfg.curriculum.pop(k, None)

    # Run-2 balance (run 1 collapsed to standing still by iter 250): rotation
    # credit doubled, body-contact penalty confined to the box top, and the
    # action-rate tax started an order of magnitude lower — with the crash
    # penalty gone from the mat, thrashing was the only thing left that made
    # an attempt cost more than doing nothing. The curriculum still raises it
    # later, once there is a flip to polish.
    cfg.rewards["flip_progress"] = RewardTermCfg(func=boxflip_mdp.flip_progress, weight=16.0)
    cfg.rewards["flip_landing"] = RewardTermCfg(func=boxflip_mdp.flip_landing, weight=4.0)
    cfg.rewards["flip_height"] = RewardTermCfg(
        func=boxflip_mdp.flip_height, weight=2.0, params={"mat_top": MAT_THICKNESS, "stand_z": 0.115}
    )
    cfg.rewards["flip_still"] = RewardTermCfg(func=boxflip_mdp.flip_still, weight=3.0)
    # v3 (2026-09-10, user rule): nothing but the feet may touch the mat, so
    # the body-contact tax applies everywhere again, and on top of it any head
    # or non-foot mat contact latches the episode dirty (boxflip_mdp._clean),
    # zeroing every landing reward. v2's final policy landed head-first for
    # ~40 ms at ~290° then rocked onto the feet; this run fine-tunes from it.
    cfg.rewards["flip_body_on_terrain"] = RewardTermCfg(
        func=boxflip_mdp.flip_body_on_terrain, weight=-1.0, params={"above_box_only": False}
    )
    # Head-tripod fix (boxflip-a100 run, 2026-09-09): that run "landed" every
    # flip with the head planted on the mat behind and the legs folded in front
    # — trunk upright at standing height, feet in contact, so feet×upright paid
    # in full. The landing terms now carry a HOME-pose Gaussian over all 14
    # servos and a head-free factor (boxflip_mdp._landed); this is the direct
    # tax on head contact, which a ballistic flip never legitimately has.
    cfg.rewards["flip_head_on_terrain"] = RewardTermCfg(func=boxflip_mdp.flip_head_on_terrain, weight=-4.0)
    # v4: feet-first touchdown bonus (one-time, per episode) — pays for the
    # rotation completed before the first mat contact. Weight is per-step;
    # 25 × one step ≈ 0.14/s averaged over the 3.5 s episode at full quality.
    cfg.rewards["flip_first_contact"] = RewardTermCfg(func=boxflip_mdp.flip_first_contact, weight=25.0)
    cfg.rewards["flip_overrotation"] = RewardTermCfg(func=boxflip_mdp.flip_overrotation, weight=-1.0)
    cfg.rewards["flip_sagittal"] = RewardTermCfg(func=boxflip_mdp.flip_sagittal, weight=-0.02)
    cfg.rewards["action_rate_l2"].weight = -0.01
    if "action_rate_weight" in cfg.curriculum:
        cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
            {"step": 0, "weight": -0.01},
            {"step": 2000 * 24, "weight": -0.05},
            {"step": 4000 * 24, "weight": -0.1},
        ]

    # Critic-sensor NaN guards, ported from the velocity env. The roulade base
    # only has the joint/root `nan_state` termination; the critic's contact and
    # air-time terms read sensor data MuJoCo can return non-finite for while
    # the state is clean, and rsl_rl's check_nan then kills the whole run.
    # That is what ended boxflip-a100 at iter 836 (2026-09-09). Critic-only,
    # so the policy is unchanged.
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_height", microduck_mdp.foot_height_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    return cfg


MicroduckBoxFlipRlCfg = replace(
    MicroduckRouladeRlCfg,
    experiment_name="microduck_boxflip",
    run_name="microduck_boxflip",
)
