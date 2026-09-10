"""N ducks side by side on one long box, one camera, one sim."""
import sys
from dataclasses import asdict
import numpy as np, torch, mediapy
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab_microduck.tasks.boxflip_terrain import BoxFlipTerrainCfg

task, ckpt, out, nsteps, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
az, el, dist = float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8])
SPACING = 0.35  # m between ducks
configure_torch_backends(); torch.manual_seed(0); np.random.seed(0)
env_cfg = load_env_cfg(task, play=True); agent_cfg = load_rl_cfg(task)
env_cfg.scene.num_envs = n; env_cfg.seed = 0
env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=(5.0, SPACING), curriculum=False, num_rows=1, num_cols=n,
    difficulty_range=(0.0, 0.0), sub_terrains={"box_flip": BoxFlipTerrainCfg()})
env_cfg.scene.terrain.max_init_terrain_level = 0
env_cfg.terminations.pop("out_of_terrain_bounds", None)  # outer tiles sit past the bound
v = env_cfg.viewer
v.env_idx = n // 2; v.max_extra_envs = n; v.distance = dist; v.elevation = el; v.azimuth = az
v.width, v.height = 1280, 720
v.origin_type = v.OriginType.WORLD
LOOKAT = tuple(float(x) for x in sys.argv[9].split(",")) if len(sys.argv) > 9 else (0.0, 0.0, 0.5)
v.lookat = LOOKAT
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode="rgb_array"), clip_actions=agent_cfg.clip_actions)
u = env.unwrapped
print("origins:", [[round(float(c), 2) for c in o] for o in u.scene.env_origins[:, :3]])
runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cpu")
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cpu")
policy = runner.get_inference_policy(device="cpu")
obs, _ = env.reset(); frames = []
with torch.no_grad():
    for t in range(nsteps):
        frames.append(u.render()); obs, *_ = env.step(policy(obs))
mediapy.write_video(out, frames, fps=50); print("wrote", out)
