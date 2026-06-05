# SmolVLA PandaStack-v1 state32 smoke training

This folder stores compact evidence for the PandaStack-v1 SmolVLA smoke training experiment.

Pipeline:
- Official teacher: sb3/tqc-PandaStack-v1
- Legacy environment: PandaStack-v1 with TimeFeatureWrapper
- Raw dataset: 20 episodes, 2000 frames
- LeRobotDataset variant: state32
- State: 32D TimeFeatureWrapper observation only
- Action: 6D padded action, where PandaStack-v1 uses real action dimensions 0..3 and padding dimensions 4..5
- SmolVLA base model: lerobot/smolvla_base
- Training steps: 50
- Batch size: 2

Validated:
- LeRobotDataset state32 loads successfully.
- SmolVLA smoke training reaches step 50.
- Checkpoints are created at steps 25 and 50.
- The trained checkpoint loads and predicts actions from dataset samples.

Offline comparison result:
- The step-50 checkpoint is only a smoke test.
- It does not improve meaningfully over the base model.
- It does not prove PandaStack-v1 task success.
- The teacher itself was previously evaluated as weak, so rollout success is not expected from this experiment alone.

Conclusion:
This experiment validates the technical PandaStack-v1 to SmolVLA training pipeline, but it does not validate a successful PandaStack policy.
