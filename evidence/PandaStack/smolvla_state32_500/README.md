# SmolVLA PandaStack-v1 state32 500-step training

This folder stores compact evidence for the PandaStack-v1 SmolVLA 500-step training experiment.

Pipeline:
- Official teacher: sb3/tqc-PandaStack-v1
- Raw dataset: 20 episodes, 2000 frames
- LeRobotDataset variant: state32
- State: 32D TimeFeatureWrapper observation only
- Action: 6D padded action
- PandaStack-v1 real action dimensions: 0..3
- Padding dimensions: 4..5
- SmolVLA base model: lerobot/smolvla_base
- Training steps: 500
- Batch size: 2

Validated:
- Training reached step 500.
- Checkpoints were created at steps 100, 200, 300, 400 and 500.
- Offline checkpoint comparison was run against the PandaStack-v1 TQC teacher.
- Checkpoint 000500 was the best checkpoint by L2 distance on real action dimensions 0..3.

Best offline checkpoint:
- checkpoint: 000500
- base_l2_all_6d: 0.102292
- checkpoint_l2_all_6d: 0.097148
- improvement_all_6d: 0.005143
- checkpoint_l2_real_0_3: 0.097148
- improvement_real_0_3: 0.005143

Conclusion:
The 500-step SmolVLA checkpoint improves offline imitation of the PandaStack-v1 TQC teacher compared with the base model. This validates a stronger training pipeline than the 50-step smoke test. It still does not prove PandaStack-v1 rollout success, especially because the official teacher was previously evaluated as weak.
