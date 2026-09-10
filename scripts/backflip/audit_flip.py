"""Whole-episode audit of the user's rules: no head contact ever, no non-foot contact
on the mat ever, feet down + head over feet at the end."""
import sys, math
from dataclasses import asdict
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab_microduck.tasks import boxflip_mdp as B
task, ckpt, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
configure_torch_backends()
env_cfg = load_env_cfg(task, play=True); agent_cfg = load_rl_cfg(task); env_cfg.scene.num_envs = N
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device="cpu"), clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cpu")
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cpu")
policy = runner.get_inference_policy(device="cpu")
u = env.unwrapped; robot = u.scene["robot"]
lf = robot.site_names.index("left_foot"); rf = robot.site_names.index("right_foot")
head_id = robot.body_names.index("jaw_soft")
obs, _ = env.reset()
box_top = u.scene.env_origins[:, 2]
head_ever = torch.zeros(N, dtype=torch.bool); body_on_mat_steps = torch.zeros(N); off_box = torch.zeros(N, dtype=torch.bool)
airborne_steps = torch.zeros(N)
with torch.no_grad():
    for t in range(170):
        obs, *_ = env.step(policy(obs))
        head_ever |= (1 - B._head_free(u)).bool()
        below = robot.data.root_link_pos_w[:, 2] < box_top - 0.05
        off_box |= below
        body_on_mat_steps += B._body_on_terrain(u) * below.float()
        # airborne = no robot contact at all
        from mjlab_microduck.tasks.mdp import _sensor_any_contact
        rc = _sensor_any_contact(u, B.ROBOT_SENSOR); airborne_steps += (~rc).float()
feet_xy = 0.5 * (robot.data.site_pos_w[:, lf, :2] + robot.data.site_pos_w[:, rf, :2])
head_xy = robot.data.body_link_pos_w[:, head_id, :2]
head_off = (head_xy - feet_xy).norm(dim=-1)
rot = torch.rad2deg(u._flip_max)
print(f"{N} episodes, checkpoint {ckpt}")
print(f"  full rotation (>=300deg)        : {(rot>=300).sum().item()}/{N}   mean {rot.mean():.0f} deg")
print(f"  reached the mat                 : {off_box.sum().item()}/{N}")
print(f"  head touched terrain EVER       : {head_ever.sum().item()}/{N}")
print(f"  non-foot contact on mat (steps) : mean {body_on_mat_steps.mean():.1f}, episodes with any: {(body_on_mat_steps>0).sum().item()}/{N}")
print(f"  airborne steps (no contact)     : mean {airborne_steps.mean():.1f} steps = {airborne_steps.mean()*0.02*1000:.0f} ms")
print(f"  END feet contact                : {B._feet(u).mean():.2f}")
print(f"  END upright                     : {B._upright(robot).mean():.2f}")
print(f"  END pose score                  : {B._pose(u, robot).mean():.2f}")
print(f"  END head-over-feet offset       : mean {head_off.mean()*100:.1f} cm, max {head_off.max()*100:.1f} cm")
