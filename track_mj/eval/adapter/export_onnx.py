import functools
import json
import os
from dataclasses import dataclass

from absl import logging
import tyro

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import track_mj as tmj
from track_mj.envs.g1_tracking.utils.wrapper import wrap_fn
from track_mj.eval.adapter.brax2onnx import (
    convert_jax2onnx,
    convert_jax2onnx_with_history,
    get_latest_ckpt,
)
from track_mj.learning.policy.model_based_ppo import train_model_based_ppo as mbppo
from track_mj.learning.policy.model_based_ppo.model_based_ppo_networks import make_model_based_ppo_networks
from track_mj.learning.policy.ppo import train_tracking as ppo
from brax.training.agents.ppo.networks import make_ppo_networks


@dataclass
class Args:
    task: str
    exp_name: str


def main(args: Args):
    ckpt_path = tmj.constant.WANDB_PATH_LOG / "adapter" / args.exp_name / "checkpoints"
    latest_ckpt = get_latest_ckpt(ckpt_path)
    if latest_ckpt is None:
        raise FileNotFoundError(f"No checkpoint found under: {ckpt_path}")

    logging.info(f"Using checkpoint: {latest_ckpt}")
    output_path = f"{latest_ckpt}/policy.onnx"

    env_class = tmj.registry.get(args.task, "tracking_adapter_train_env_class")
    task_cfg = tmj.registry.get(args.task, "tracking_adapter_config")
    env_cfg = task_cfg.env_config
    policy_config = task_cfg.policy_config
    mbppo_policy_config = getattr(task_cfg, "mbppo_policy_config", None)

    config_path = ckpt_path / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    config["policy_config"].pop("progress_fn", None)
    env_cfg.update(config["env_config"])
    policy_config.update(config["policy_config"])
    if mbppo_policy_config is not None and "mbppo_policy_config" in config:
        mbppo_policy_config.update(config["mbppo_policy_config"])

    env = env_class(terrain_type=env_cfg.terrain_type, config=env_cfg)
    env.prepare_trajectory(env._config.reference_traj_config.name)

    use_mbppo = (
        mbppo_policy_config is not None
        and (bool(mbppo_policy_config.use_adapter) or bool(mbppo_policy_config.train_world_model))
    )

    if use_mbppo:
        network_factory = functools.partial(
            make_model_based_ppo_networks,
            **policy_config.network_factory,
            **mbppo_policy_config.network_factory,
        )
        train_fn = functools.partial(
            mbppo.train,
            num_timesteps=0,
            episode_length=policy_config.episode_length,
            normalize_observations=False,
            restore_checkpoint_path=latest_ckpt,
            network_factory=network_factory,
            wrap_env_fn=wrap_fn,
            num_envs=1,
            **mbppo_policy_config,
        )
    else:
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

    obs_size = env.observation_size
    act_size = env.action_size

    if use_mbppo:
        convert_jax2onnx_with_history(
            output_path=output_path,
            inference_fn=inference_fn,
            policy_network_cfg=policy_config.network_factory,
            mbppo_network_cfg=mbppo_policy_config.network_factory,
            obs_size=obs_size,
            action_size=act_size,
            history_len=env_cfg.history_len,
            jax_params=params,
            use_adapter=mbppo_policy_config.use_adapter,
            activation="swish",
        )
    else:
        convert_jax2onnx(
            ckpt_dir=latest_ckpt,
            output_path=output_path,
            inference_fn=inference_fn,
            hidden_layer_sizes=policy_config.network_factory.policy_hidden_layer_sizes,
            obs_size=obs_size,
            action_size=act_size,
            policy_obs_key=policy_config.network_factory.policy_obs_key,
            jax_params=params,
            activation="swish",
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
