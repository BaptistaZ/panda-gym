# SmolVLA PandaPickAndPlace 500-step Training

Training run for validating short fine-tuning of SmolVLA on the PandaPickAndPlace TQC teacher dataset.

## Dataset

- Repo ID: `local/panda_pickandplace_tqc_teacher_100`
- Root: `outputs/lerobot_datasets/panda_pickandplace_tqc_teacher_100`
- Episodes: 100
- Frames: 751
- Task: `pick and place the cube`
- Observation state shape: 25
- Action shape: 6
- Cameras: 3 RGB views, each `[3, 256, 256]`

## Training

- Policy: `smolvla`
- Pretrained path: `lerobot/smolvla_base`
- Device: CUDA
- Steps: 500
- Batch size: 2
- Checkpoint saved at step 500
- Final logged loss: 0.775
- Training status: completed

## Checkpoint

Local checkpoint:

`outputs/train/smolvla_panda_pickandplace_tqc_100_500/checkpoints/000500/pretrained_model`

The full training output is stored locally under `outputs/train/` and is not versioned in Git.
