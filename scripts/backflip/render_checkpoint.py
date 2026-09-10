"""Headless render of a checkpoint: loads env with rgb_array, runs the policy, writes mp4."""
import sys, math
from dataclasses import asdict
import numpy as np, torch, mediapy
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

task, ckpt, out, nsteps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
configure_torch_backends()
device = "cpu"
env_cfg = load_env_cfg(task, play=True)
agent_cfg = load_rl_cfg(task)
env_cfg.scene.num_envs = 1
env_cfg.viewer.distance = 1.2
env_cfg.viewer.elevation = -10.0
env_cfg.viewer.azimuth = 90.0   # side view, so the sagittal roll reads clearly
env_cfg.viewer.width, env_cfg.viewer.height = 960, 540
# spawn from standing only (no mid-roll spawns) so we test the full move
for name, ev in env_cfg.events.items():
    if "spawn" in name or "roulade_state" in name:
        print("[event]", name, ev.params)
env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
runner = runner_cls(env, asdict(agent_cfg), device=device)
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

obs, _ = env.reset()
frames, rot = [], []
u = env.unwrapped
robot = u.scene["robot"]
with torch.no_grad():
    for t in range(nsteps):
        frames.append(env.unwrapped.render())
        act = policy(obs)
        obs, rew, dones, extras = env.step(act)
        g = robot.data.projected_gravity_b[0]
        z = robot.data.root_link_pos_w[0, 2].item()
        rot.append((t, z, g[2].item()))
        if t % 25 == 0:
            print(f"t={t/50:.2f}s z={z:.3f} upright={-g[2].item():+.2f}")
mediapy.write_video(out, frames, fps=50)
print("wrote", out, len(frames), "frames")
