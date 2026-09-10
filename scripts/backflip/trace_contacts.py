import sys
from dataclasses import asdict
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab_microduck.tasks import boxflip_mdp as B
from mjlab_microduck.tasks.mdp import _sensor_any_contact
task, ckpt = sys.argv[1], sys.argv[2]
configure_torch_backends()
env_cfg = load_env_cfg(task, play=True); agent_cfg = load_rl_cfg(task); env_cfg.scene.num_envs = 1
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device="cpu"), clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cpu")
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cpu")
policy = runner.get_inference_policy(device="cpu")
u = env.unwrapped; robot = u.scene["robot"]
lf = robot.site_names.index("left_foot"); rf = robot.site_names.index("right_foot"); hid = robot.body_names.index("jaw_soft")
obs, _ = env.reset()
def hoff():
    f = 0.5*(robot.data.site_pos_w[0,lf,:2]+robot.data.site_pos_w[0,rf,:2]); return (robot.data.body_link_pos_w[0,hid,:2]-f).norm().item()*100
print(f"t=0 HOME standing: head-over-feet offset = {hoff():.1f} cm, head z above trunk = {(robot.data.body_link_pos_w[0,hid,2]-robot.data.root_link_pos_w[0,2]).item()*100:.1f} cm")
box_top = u.scene.env_origins[0,2].item()
with torch.no_grad():
    for t in range(90):
        obs, *_ = env.step(policy(obs))
        feet = B._feet(u)[0].item(); head = 1-B._head_free(u)[0].item(); rc = _sensor_any_contact(u, B.ROBOT_SENSOR)[0].item()
        z = robot.data.root_link_pos_w[0,2].item()
        where = "box" if z > box_top-0.05 else "mat"
        flag = ""
        if head: flag += " HEAD"
        if rc and not feet: flag += " NONFOOT"
        if not rc: flag += " air"
        if flag.strip() != "air" and flag or t%10==0:
            print(f"t={t*20:4d}ms {where} z={z:.3f} feet={int(feet)} rot={torch.rad2deg(u._flip_max[0]):4.0f}{flag}")
print(f"END head-over-feet offset = {hoff():.1f} cm")
