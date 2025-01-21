# Motion-based Human Control (MHC)

## Table of Contents
- [Installation](#installation)
- [Motion Data](#motion-data)
- [Training and Evaluation](#training-and-evaluation)
  - [SMPL Model](#smpl-model)
  - [Reallusion Model](#reallusion-model)

## Installation

- Download from the [NVIDIA Isaac Gym website](https://developer.nvidia.com/isaac-gym)
- Follow one of these installation methods:

**Method A**: Simple Installation
```bash
pip install -r requirements.txt
```

**Method B**: Detailed Installation
```bash
# Create and activate environment
conda create -n mhc python=3.8
conda activate mhc

# Install PyTorch and dependencies
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 \
        tensorboard==2.8.0 torchaudio==0.10.0+cu113 \
        -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install Isaac Gym
cd isaacgym/python && pip install -e .

# Install MHC
cd mhc
pip install -e .
```

## Motion Data

### Location and Format
- Motion clips are stored in `ase/data/motions/` as `.npy` files
- Datasets are defined in `.yaml` files containing lists of motion clips

### Visualization
To visualize motion clips, use:
```bash
python ase/run.py --test \
    --task HumanoidViewMotion \
    --num_envs 2 \
    --cfg_env ase/data/cfg/humanoid_sword_shield.yaml \
    --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml \
    --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Atk_2xCombo01_Motion.npy
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