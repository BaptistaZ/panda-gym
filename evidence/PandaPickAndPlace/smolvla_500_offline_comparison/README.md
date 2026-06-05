# SmolVLA PandaPickAndPlace 500-step Offline Comparison

Offline comparison between the SmolVLA base model and the 500-step fine-tuned checkpoint on the PandaPickAndPlace TQC teacher dataset.

## Dataset

- Task: `PandaPickAndPlace-v3`
- Dataset: `local/panda_pickandplace_tqc_teacher_100`
- Frames: 751
- Episodes: 100
- Teacher: `chencliu/tqc-PandaPickAndPlace-v3`

## Compared models

- Base model: `lerobot/smolvla_base`
- Fine-tuned checkpoint: `outputs/train/smolvla_panda_pickandplace_tqc_100_500/checkpoints/000500/pretrained_model`

## Main result

On the full dataset:

- Base mean L2 real 4D: 1.9214
- Checkpoint mean L2 real 4D: 1.3283
- Mean improvement: 0.5931
- Relative improvement: approximately 30.9%

The 4 real action dimensions are `[dx, dy, dz, gripper]`. The remaining two action dimensions are padding for SmolVLA compatibility.

This is an offline imitation metric only. It does not prove rollout success in the environment.
