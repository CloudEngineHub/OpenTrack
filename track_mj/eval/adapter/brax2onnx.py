from typing import Dict, Union, Tuple, Mapping
import functools
from absl import logging
from dataclasses import dataclass
import tyro

# --- Set environment variables ---
import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# --- TensorFlow GPU setup ---
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.keras.mixed_precision.set_global_policy("float32")

import numpy as np
import matplotlib.pyplot as plt
import jax
import tf2onnx
import onnxruntime as rt
import torch


# --- MLP model definition ---
class MLP(tf.keras.Model):
    def __init__(
        self,
        layer_sizes,
        activation=tf.nn.relu,
        kernel_init="lecun_uniform",
        activate_final=False,
        bias=True,
        layer_norm=False,
        use_tanh_distribution=True,
    ):
        super().__init__()
        self.activation = activation
        self.activate_final = activate_final
        self.layer_norm = layer_norm
        self.model = tf.keras.Sequential(name="MLP_0")
        self.use_tanh_distribution = use_tanh_distribution

        for i, size in enumerate(layer_sizes):
            self.model.add(
                tf.keras.layers.Dense(
                    size, activation=None, use_bias=bias, kernel_initializer=kernel_init, name=f"hidden_{i}"
                )
            )
            if i != len(layer_sizes) - 1 or activate_final:
                if layer_norm:
                    self.model.add(tf.keras.layers.LayerNormalization(name=f"ln_{i}"))

    def call(self, inputs):
        x = inputs
        for layer in self.model.layers:
            x = layer(x)
            if isinstance(layer, tf.keras.layers.Dense):
                if self.activate_final or not layer.name.endswith(
                    f"{len(self.model.layers) // (2 if self.layer_norm else 1) - 1}"
                ):
                    x = self.activation(x)
        loc, _ = tf.split(x, 2, axis=-1)
        if self.use_tanh_distribution:
            return tf.tanh(loc)
        else:
            return loc


# --- Utility functions ---
def build_tf_policy_network(
    action_size,
    hidden_layer_sizes,
    activation="swish",
    kernel_init="lecun_uniform",
    layer_norm=False,
    use_tanh_distribution=True,
):
    if activation == "swish":
        activation = tf.nn.swish
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

    return MLP(
        layer_sizes=list(hidden_layer_sizes) + [action_size * 2],
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
        use_tanh_distribution=use_tanh_distribution,
    )


def transfer_weights(jax_params, tf_model):
    for name, params in jax_params.items():
        try:
            tf_layer = tf_model.get_layer("MLP_0").get_layer(name=name)
        except ValueError:
            logging.error(f"Layer {name} not found in TF model.")
            continue
        if isinstance(tf_layer, tf.keras.layers.Dense):
            tf_layer.set_weights([np.array(params["kernel"]), np.array(params["bias"])])
        else:
            logging.error(f"Unhandled layer type: {type(tf_layer)}")
    logging.info("Weights transferred successfully.")


class TorchMLP(torch.nn.Module):
    def __init__(self, layer_sizes, activation="swish", activate_final=False, bias=True, split=False):
        super().__init__()
        self.act = torch.nn.SiLU() if activation == "swish" else torch.nn.ReLU()
        self.activate_final = activate_final
        self.split = split

        hidden = []
        for idx in range(len(layer_sizes) - 1):
            hidden.append(torch.nn.Linear(layer_sizes[idx], layer_sizes[idx + 1], bias=bias))
        self.hidden = torch.nn.ModuleList(hidden)

    def forward(self, x):
        for i, layer in enumerate(self.hidden):
            x = layer(x)
            if i != len(self.hidden) - 1 or self.activate_final:
                x = self.act(x)
        if self.split:
            loc, _ = torch.chunk(x, 2, dim=-1)
            return torch.tanh(loc)
        return x


class TorchMLPWithAdapter(torch.nn.Module):
    def __init__(self, layer_sizes, embedding_size, activation="swish", activate_final=False, bias=True, split=False):
        super().__init__()
        self.act = torch.nn.SiLU() if activation == "swish" else torch.nn.ReLU()
        self.activate_final = activate_final
        self.split = split

        hidden = []
        adapter = []
        for idx in range(len(layer_sizes) - 1):
            in_dim, out_dim = layer_sizes[idx], layer_sizes[idx + 1]
            hidden.append(torch.nn.Linear(in_dim, out_dim, bias=bias))
            adapter.append(torch.nn.Linear(embedding_size if idx == 0 else in_dim, out_dim, bias=bias))
        self.hidden = torch.nn.ModuleList(hidden)
        self.adapter = torch.nn.ModuleList(adapter)

    def forward(self, x, emb):
        base_hidden = self.hidden[0](x)
        adapter_hidden = self.adapter[0](emb)
        for i in range(1, len(self.hidden)):
            out = self.act(base_hidden + adapter_hidden)
            base_hidden = self.hidden[i](out)
            adapter_hidden = self.adapter[i](out)
        out = base_hidden + adapter_hidden
        if self.split:
            loc, _ = torch.chunk(out, 2, dim=-1)
            return torch.tanh(loc)
        return out


class TorchConvMLP(torch.nn.Module):
    def __init__(self, num_filters, kernel_sizes, strides, history_len, layer_sizes, activation="swish", bias=True):
        super().__init__()
        self.act = torch.nn.SiLU() if activation == "swish" else torch.nn.ReLU()
        self.history_len = history_len

        conv = []
        for idx in range(len(num_filters) - 1):
            conv.append(torch.nn.Conv1d(num_filters[idx], num_filters[idx + 1], kernel_sizes[idx], strides[idx], padding="valid"))
        self.conv = torch.nn.ModuleList(conv)

        hidden = []
        for idx in range(len(layer_sizes) - 1):
            hidden.append(torch.nn.Linear(layer_sizes[idx], layer_sizes[idx + 1], bias=bias))
        self.hidden = torch.nn.ModuleList(hidden)

    def forward(self, x):
        for layer in self.conv:
            x = self.act(layer(x))
        x = x.transpose(-1, -2).reshape(*x.shape[:-2], -1)
        for i, layer in enumerate(self.hidden):
            x = layer(x)
            if i != len(self.hidden) - 1:
                x = self.act(x)
        return x


class TorchCombinedModel(torch.nn.Module):
    def __init__(self, policy_model: torch.nn.Module, history_model: torch.nn.Module, use_adapter: bool):
        super().__init__()
        self.policy_model = policy_model
        self.history_model = history_model
        self.use_adapter = use_adapter

    def forward(self, obs, history):
        history_embed = self.history_model(history)
        if self.use_adapter:
            return self.policy_model(obs, history_embed)
        return self.policy_model(torch.cat([obs, history_embed], dim=-1))


def transfer_weights_to_torch(jax_params: Mapping, torch_model: torch.nn.Module):
    for name, params in jax_params.items():
        if name.startswith("hidden_"):
            layer = torch_model.hidden[int(name.split("_")[-1])]
        elif name.startswith("adapter_"):
            layer = torch_model.adapter[int(name.split("_")[-1])]
        elif name.startswith("conv_"):
            layer = torch_model.conv[int(name.split("_")[-1])]
        else:
            raise ValueError(f"Unexpected parameter name: {name}")

        layer.weight.data[:] = torch.tensor(np.array(params["kernel"]).T, dtype=torch.float32)
        layer.bias.data[:] = torch.tensor(np.array(params["bias"]), dtype=torch.float32)

    logging.info("Weights transferred (JAX -> Torch) successfully.")


def get_latest_ckpt(path):
    from pathlib import Path

    ckpts = [ckpt for ckpt in Path(path).glob("*") if not ckpt.name.endswith(".json")]
    ckpts.sort(key=lambda x: int(x.name))
    return ckpts[-1] if ckpts else None


def convert_jax2onnx(
    ckpt_dir,
    output_path,
    inference_fn,
    hidden_layer_sizes,
    obs_size: Union[int, Mapping[str, Union[Tuple[int, ...], int]]],
    action_size: int,
    policy_obs_key,
    jax_params,
    activation="swish",
    use_tanh_distribution=True,
):
    rand_obs = {
        "state": np.random.randn(1, obs_size["state"][0]).astype(np.float32),
        "privileged_state": np.random.randn(1, obs_size["privileged_state"][0]).astype(np.float32),
    }

    jax_pred, _ = inference_fn(rand_obs, jax.random.PRNGKey(0))
    jax_pred = np.array(jax_pred[0])

    tf_model = build_tf_policy_network(
        action_size=action_size,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        use_tanh_distribution=use_tanh_distribution,
    )

    example_input = tf.ones((1, obs_size[policy_obs_key][0]))
    tf_model(example_input)  # build model

    transfer_weights(jax_params[1]["params"], tf_model)

    test_input = [rand_obs[policy_obs_key].reshape(1, -1)]
    tf_pred = tf_model(test_input)[0][0].numpy()

    tf_model.output_names = ["continuous_actions"]

    # Dynamic shape for ONNX conversion
    spec = (tf.TensorSpec([None, obs_size[policy_obs_key][0]], tf.float32, name="obs"),)
    tf2onnx.convert.from_keras(
        tf_model, input_signature=spec, opset=11, output_path=output_path
    )

    sess = rt.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    onnx_pred = sess.run(["continuous_actions"], {"obs": test_input[0].astype(np.float32)})[0][0]

    logging.info("Predictions:")
    np.set_printoptions(precision=2, suppress=True)
    logging.info(f"\n\tJAX  : {jax_pred}\n\tTF   : {tf_pred}\n\tONNX : {onnx_pred}")
    jax2onnx_mae = np.mean(np.abs(jax_pred - onnx_pred))

    np.testing.assert_allclose(onnx_pred, tf_pred, rtol=1e-03, atol=1e-05)
    logging.info(f"Mean absolute error: {jax2onnx_mae:.2e}")
    logging.info(f"Success! ONNX model saved to {output_path}")


def convert_jax2onnx_with_history(
    output_path,
    inference_fn,
    policy_network_cfg,
    mbppo_network_cfg,
    obs_size: Mapping[str, Union[Tuple[int, ...], int]],
    action_size: int,
    history_len: int,
    jax_params,
    use_adapter: bool = True,
    activation: str = "swish",
    onnx_opset_version: int = 18,
):
    def _flat_obs_dim(size):
        if isinstance(size, int):
            return size
        if isinstance(size, tuple):
            prod = 1
            for dim in size:
                prod *= dim
            return prod
        # Fallback for list-like shapes
        prod = 1
        for dim in size:
            prod *= dim
        return prod

    if history_len <= 0:
        raise ValueError(f"history_len must be positive for MBPPO export, got {history_len}")

    # JAX policy expects flattened history_state in the same format as environment observation.
    rand_obs = {key: np.random.randn(1, _flat_obs_dim(obs_size[key])).astype(np.float32) for key in obs_size.keys()}
    jax_pred, _ = inference_fn(rand_obs, jax.random.PRNGKey(0))
    jax_pred = np.array(jax_pred[0])

    history_flat_dim = rand_obs["history_state"].shape[-1]
    if history_flat_dim % history_len != 0:
        raise ValueError(
            f"history_state dim {history_flat_dim} is not divisible by history_len {history_len}."
        )
    history_dim = history_flat_dim // history_len
    history_for_onnx = rand_obs["history_state"].reshape(1, history_len, history_dim).swapaxes(-1, -2)
    obs_dim = obs_size[policy_network_cfg.policy_obs_key][0]
    embedding_size = mbppo_network_cfg.embedding_size

    if use_adapter:
        policy_layer_sizes = [obs_dim] + list(policy_network_cfg.policy_hidden_layer_sizes) + [action_size * 2]
        policy_model = TorchMLPWithAdapter(policy_layer_sizes, embedding_size, activation=activation, split=True)
    else:
        policy_layer_sizes = [obs_dim + embedding_size] + list(policy_network_cfg.policy_hidden_layer_sizes) + [action_size * 2]
        policy_model = TorchMLP(policy_layer_sizes, activation=activation, split=True)
    transfer_weights_to_torch(jax_params[1]["params"], policy_model)
    policy_model.eval()

    history_len_after_conv = history_len
    for kernel_size, stride in zip(mbppo_network_cfg.history_encoder_kernel_sizes, mbppo_network_cfg.history_encoder_strides):
        history_len_after_conv = (history_len_after_conv - kernel_size[0]) // stride[0] + 1

    num_filters = [history_dim] + list(mbppo_network_cfg.history_encoder_num_filters)
    history_layer_sizes = [num_filters[-1] * history_len_after_conv] + list(mbppo_network_cfg.history_encoder_hidden_layer_sizes) + [embedding_size]
    history_model = TorchConvMLP(
        num_filters=num_filters,
        kernel_sizes=mbppo_network_cfg.history_encoder_kernel_sizes,
        strides=mbppo_network_cfg.history_encoder_strides,
        history_len=history_len,
        layer_sizes=history_layer_sizes,
        activation=activation,
    )
    transfer_weights_to_torch(jax_params[4]["params"], history_model)
    history_model.eval()

    torch_model = TorchCombinedModel(policy_model, history_model, use_adapter)
    torch_model.eval()

    with torch.no_grad():
        _ = torch_model(
            torch.from_numpy(rand_obs[policy_network_cfg.policy_obs_key]),
            torch.from_numpy(history_for_onnx),
        )

    dummy_history = torch.ones(1, history_dim, history_len, dtype=torch.float32)
    dummy_obs = torch.ones(1, obs_size[policy_network_cfg.policy_obs_key][0], dtype=torch.float32)
    torch.onnx.export(
        torch_model,
        (dummy_obs, dummy_history),
        output_path,
        input_names=["obs", "history"],
        output_names=["continuous_actions"],
        dynamic_axes={"obs": {0: "batch"}, "history": {0: "batch"}, "continuous_actions": {0: "batch"}},
        opset_version=onnx_opset_version,
    )

    sess = rt.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    onnx_pred = sess.run(
        ["continuous_actions"],
        {
            "obs": rand_obs[policy_network_cfg.policy_obs_key].astype(np.float32),
            "history": history_for_onnx.astype(np.float32),
        },
    )[0][0]
    jax2onnx_mae = np.mean(np.abs(jax_pred - onnx_pred))
    logging.info(f"[MBPPO] Mean absolute error: {jax2onnx_mae:.2e}")
    logging.info(f"[MBPPO] Success! ONNX model saved to {output_path}")