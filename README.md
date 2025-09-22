<div align="center">
  <h1 align="center"> OpenTrack </h1>
  <h3 align="center"> GALBOT · Tsinghua </h3>
<!--   <p align="center">
    <a href="README.md"> English </a> | <a href="README_zh.md">中文</a>
  </p>     -->

:page_with_curl:[Paper](https://arxiv.org/abs/2509.13833) | :house:[Website](https://zzk273.github.io/Any2Track/)


This repository is the official implementation of OpenTrack, an open-source humanoid motion tracking codebase that uses MuJoCo for simulation and supports multi-GPU parallel training.
</div>

# Prepare

1. Clone the repository:
   ```shell
   git clone git@github.com:GalaxyGeneralRobotics/OpenTrack.git
   ```

2. Create a virtual environment and install dependencies:
   ```shell
   conda create -n any2track python=3.12
   conda activate any2track
   # Install torch to convert JAX to Torch. We don't require the GPU version of torch, but you can install any version as you like.
   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

5. Download the [mocap data](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main/Lafan1/mocap/UnitreeG1) and put them under `data/mocap/`. Thanks for the retargeting motions of LAFAN1 dataset from [LocoMuJoCo](https://github.com/robfiras/loco-mujoco/)!

   The file structure should be like:

   ```
   data/
   |-- xmls
      |- ...
   |-- mocap
      |-- lafan1
         |-- UnitreeG1
               |-- dance1_subject1.npz
               |--- ...
   ```

## Usage ##

1. Train the model
   ```shell
   # Train on a flat terrain:
   python train_policy.py --exp_name flat_terrain --terrain_type flat_terrain
   # Train on a rough terrain:
   python generate_terrain.py # generate various hfield with Perlin noise
   python train_policy.py --exp_name rough_terrain --terrain_type rough_terrain
   
   # For debug mode (quick testing training without logging)
   # python train_policy.py --exp_name debug 
   ```

2. Evaluate the model
   First, convert the Brax model checkpoint to PyTorch:
   ```shell
   # your_exp_name=<timestamp>_<exp_name>
   python brax2torch.py --exp_name <your_exp_name>
   ```

   Next, run the evaluation script:
   
   ```shell
   # your_exp_name=<timestamp>_<exp_name>
   python play_policy.py --exp_name <your_exp_name> [--use_viewer] [--use_renderer] [---play_ref_motion]
   ```

# TODOs

- [x] Release AnyTracker
- [x] Release dynamics disturbances
- [ ] Release AnyAdapter
- [ ] Release real deployment code
   
## Acknowledgement

This repository is build upon `jax`, `brax`, `loco-mujoco`, and `mujoco_playground`.

If you find this repository helpful, please cite our work:

```bibtex
@article{zhang2025track,
  title={Track Any Motions under Any Disturbances},
  author={Zhikai Zhang and Jun Guo and Chao Chen and Jilong Wang and Chenghuai Lin and Yunrui Lian and Han Xue and Zhenrong Wang and Maoqi Liu and Huaping Liu and He Wang and Li Yi},
  journal={arXiv preprint arXiv:2509.13833},
  year={2025}
}
```