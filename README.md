# Motion-based Human Control (MHC)

## Table of Contents
- [Installation](#installation)
- [Motion Data](#motion-data)
- [Training and Evaluation](#training-and-evaluation)
  - [SMPL Model](#smpl-model)
  - [Reallusion Model](#reallusion-model)


Code accompanying the paper: "Generating Physically Realistic and Directable Human Motions from Multi-Modal Inputs"



## Installation

### Set up the conda environment
```bash
conda create -n mhc python=3.8
conda activate mhc

pip install -r requirements.txt
```

### Install Isaac Gym

Download from the [NVIDIA Isaac Gym website](https://developer.nvidia.com/isaac-gym)

Navigate to the installation directory and run:

```bash
cd isaacgym/python && pip install -e .
```

## Motion Data

### Location and Format
- Motion clips are stored in `mhc/data/motions/` as `.npy` files
- Datasets are defined in `.yaml` files containing lists of motion clips

### Visualization
To visualize motion clips, use:
```bash
python mhc/mhc_play.py --test \
    --task HumanoidViewMotion \
    --num_envs 2 \
    --cfg_env mhc/data/cfg/humanoid_mhc_sword_shield_catchup.yaml \
    --cfg_train mhc/data/cfg/train/rlg/mhc_humanoid_catchup.yaml \
    --motion_file mhc/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml
```
Note: Use `--motion_file` with either a `.npy` file (single motion) or `.yaml` file (dataset)

### Motion Data Credits
Motion data provided by Reallusion (for non-commercial use only):
- [Studio Mocap Sword and Shield Stunts](https://actorcore.reallusion.com/motion/pack/studio-mocap-sword-and-shield-stunts)
- [Studio Mocap Sword and Shield Moves](https://actorcore.reallusion.com/motion/pack/studio-mocap-sword-and-shield-moves)

For custom motion retargeting, refer to: `ase/poselib/retarget_motion.py`

## Training and Evaluation

### SMPL Model

#### Training
```bash
python mhc/mhc_train.py \
    --train_run \
    --num_envs 4096 \
    --xbox_interactive \
    --demo_motion_ids 9999 \
    --mlp_units 1024,512 \
    --disc_units 1024,512 \
    --lk_encoder_units 1024,1024 \
    --lk_embedding_dim 64 \
    --max_epochs 200000 \
    --motion_lib_device cpu \
    --cfg_env mhc/data/cfg/humanoid_mhc_smpl_catchup.yaml \
    --cfg_train mhc/data/cfg/train/rlg/mhc_humanoid_catchup.yaml \
    --motion_file mhc/data/motions/amass/amp_humanoid_amass_dataset_sfu.yaml \
    --wandb_project mhc_smpl_train \
    --wandb_path WANDB_USERNAME/mhc_smpl_train/wandbRunID/latest_model.pth \
    --disable_early_termination \
    --energy_penalty \
    --penalty_multiplyer 2 \
    --switch_demos_within_episode \
    --init_state_rotate \
    --demo_rotate
```

#### Evaluation
```bash
python mhc/mhc_play.py \
    --visualize_policy \
    --num_envs 3 \
    --interactive \
    --env_spacing 1.5 \
    [... same parameters as training ...]
```

### Reallusion Model

#### Training
```bash
python mhc/mhc_train.py \
    --train_run \
    --num_envs 4096 \
    --env_spacing 1.5 \
    --demo_motion_ids 9999 \
    --mlp_units 1024,512 \
    --disc_units 1024,512 \
    --lk_encoder_units 1024,1024 \
    --lk_embedding_dim 64 \
    --max_epochs 200000 \
    --motion_lib_device cpu \
    --cfg_env mhc/data/cfg/humanoid_mhc_sword_shield_catchup.yaml \
    --cfg_train mhc/data/cfg/train/rlg/mhc_humanoid_catchup.yaml \
    --motion_file mhc/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml \
    --wandb_project mhc_reallusion_train \
    --wandb_path WANDB_USERNAME/mhc_reallusion_train/wandbRunID/latest_model.pth \
    --disable_early_termination \
    --energy_penalty \
    --penalty_multiplyer 2 \
    --switch_demos_within_episode \
    --init_state_rotate \
    --demo_rotate
```

#### Evaluation
```bash
python mhc/mhc_play.py \
    --visualize_policy \
    --num_envs 3 \
    --interactive \
    --env_spacing 1.5 \
    [... same parameters as training ...]
```

## Citation
If you find this codebase useful for your research, please cite our paper:

```
@article{shrestha2024generating,
  title={Generating Physically Realistic and Directable Human Motions from Multi-Modal Inputs},
  author={Shrestha, Aayam and Liu, Pan and Ros, German and Yuan, Kai and Fern, Alan},
  journal={ECCV},
  year={2024}
}
```

## References
- [ASE](https://github.com/nv-tlabs/ASE) [The codebase is largely adapted from ASE]
- [Isaac Gym](https://developer.nvidia.com/isaac-gym)
- [Reallusion](https://actorcore.reallusion.com/)
- [AMASS](https://amass.is.tue.mpg.de/)
- [SMPL](https://smpl.is.tue.mpg.de/)
