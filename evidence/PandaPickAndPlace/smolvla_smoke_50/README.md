# SmolVLA PandaPickAndPlace Smoke Training

Smoke training run for validating that the PandaPickAndPlace TQC teacher dataset can be consumed by SmolVLA training.

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
- Steps: 50
- Batch size: 2
- Checkpoint saved at step 50
- Final logged loss: 0.721
- Training status: completed

The full training output is stored locally under `outputs/train/` and is not versioned in Git.
