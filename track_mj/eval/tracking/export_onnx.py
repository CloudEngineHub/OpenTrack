import functools
import json
import os
from dataclasses import dataclass

from absl import logging
import tyro

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from brax.training.agents.ppo.networks import make_ppo_networks

import track_mj as tmj
from track_mj.envs.g1_tracking.utils.wrapper import wrap_fn
from track_mj.eval.tracking.brax2onnx import convert_jax2onnx, get_latest_ckpt
from track_mj.learning.policy.ppo import train_tracking as ppo


@dataclass
class Args:
    task: str
    exp_name: str


def main(args: Args):
    ckpt_path = tmj.constant.WANDB_PATH_LOG / "track" / args.exp_name / "checkpoints"
    latest_ckpt = get_latest_ckpt(ckpt_path)
    if latest_ckpt is None:
        raise FileNotFoundError(f"No checkpoint found under: {ckpt_path}")

    logging.info(f"Using checkpoint: {latest_ckpt}")
    output_path = f"{latest_ckpt}/policy.onnx"

    env_class = tmj.registry.get(args.task, "tracking_train_env_class")
    task_cfg = tmj.registry.get(args.task, "tracking_config")
    env_cfg = task_cfg.env_config
    policy_config = task_cfg.policy_config

    config_path = ckpt_path / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    config["policy_config"].pop("progress_fn", None)
    env_cfg.update(config["env_config"])
    policy_config.update(config["policy_config"])

    env = env_class(terrain_type=env_cfg.terrain_type, config=env_cfg)
    env.prepare_trajectory(env._config.reference_traj_config.name)

    network_factory = functools.partial(make_ppo_networks, **policy_config.network_factory)
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=0,
        episode_length=policy_config.episode_length,
        normalize_observations=False,
        restore_checkpoint_path=latest_ckpt,
        network_factory=network_factory,
        wrap_env_fn=wrap_fn,
        num_envs=1,
    )

    make_inference_fn, params, _ = train_fn(environment=env)
    inference_fn = make_inference_fn(params, deterministic=True)

    convert_jax2onnx(
        ckpt_dir=latest_ckpt,
        output_path=output_path,
        inference_fn=inference_fn,
        hidden_layer_sizes=policy_config.network_factory.policy_hidden_layer_sizes,
        obs_size=env.observation_size,
        action_size=env.action_size,
        policy_obs_key=policy_config.network_factory.policy_obs_key,
        jax_params=params,
        activation="swish",
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
