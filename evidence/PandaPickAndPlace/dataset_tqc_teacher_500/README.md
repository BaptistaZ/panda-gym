# PandaPickAndPlace-v3 TQC Teacher Dataset 500

Dataset generated from the validated Hugging Face teacher:

`chencliu/tqc-PandaPickAndPlace-v3`

## Raw teacher dataset

- Episodes: 500
- Frames / total steps: 3712
- Teacher success count: 500/500
- Teacher success rate: 1.0
- State shape: 25
- Environment action shape: 4
- SmolVLA action shape: 6
- Task: `pick and place the cube`

## LeRobotDataset conversion

- Repo ID: `local/panda_pickandplace_tqc_teacher_500`
- Root: `outputs/lerobot_datasets/panda_pickandplace_tqc_teacher_500`
- Converted episodes: 500
- Converted frames: 3712
- `observation.state`: `[25]`
- `action`: `[6]`
- `observation.images.camera1/2/3`: `[3, 256, 256]`

The raw dataset and LeRobotDataset are stored locally under `outputs/` and are not versioned in Git.
