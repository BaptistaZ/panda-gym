# PandaPickAndPlace-v3 TQC Teacher Dataset

Dataset generated from the validated Hugging Face teacher:

`chencliu/tqc-PandaPickAndPlace-v3`

## Raw teacher dataset

- Episodes: 100
- Frames: 751
- Success rate: 1.0
- Success count: 100/100
- State shape: 25
- Environment action shape: 4
- SmolVLA action shape: 6

## LeRobotDataset conversion

- Repo ID: `local/panda_pickandplace_tqc_teacher_100`
- Frames: 751
- Episodes: 100
- `observation.state`: `[25]`
- `action`: `[6]`
- `observation.images.camera1/2/3`: `[3, 256, 256]`
- Task: `pick and place the cube`

The raw dataset and LeRobotDataset are stored locally under `outputs/` and are not versioned in Git.
