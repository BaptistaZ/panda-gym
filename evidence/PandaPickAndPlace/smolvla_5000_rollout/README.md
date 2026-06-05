# SmolVLA PandaPickAndPlace 5000-step Rollout

Rollout evaluation of the SmolVLA checkpoint trained for 5000 steps on the PandaPickAndPlace TQC teacher dataset.

## Checkpoint

`outputs/train/smolvla_panda_pickandplace_tqc_100_5000/checkpoints/005000/pretrained_model`

## Dataset

- Repo ID: `local/panda_pickandplace_tqc_teacher_100`
- Episodes: 100
- Frames: 751
- Teacher: `chencliu/tqc-PandaPickAndPlace-v3`

## Rollout results

### action_scale = 1.0

- Success count: 1/20
- Success rate: 0.05
- Mean reward: -49.25
- Mean distance improvement: 0.0339

### action_scale = 0.75

- Success count: 3/20
- Success rate: 0.15
- Mean reward: -44.5
- Mean distance improvement: 0.0159

### action_scale = 0.5

- Success count: 2/20
- Success rate: 0.10
- Mean reward: -46.6
- Mean distance improvement: 0.0088

## Conclusion

The 5000-step checkpoint achieved real rollout success in PandaPickAndPlace-v3. The best tested action scale was 0.75, with 3/20 successful episodes. The success rate is still low, so the next step is to generate a larger teacher dataset.
