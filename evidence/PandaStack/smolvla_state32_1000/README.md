# SmolVLA PandaStack-v1 state32 1000-step training

This folder stores compact evidence for the PandaStack-v1 SmolVLA 1000-step training experiment.

Pipeline:
- Official teacher: sb3/tqc-PandaStack-v1
- Raw dataset: 20 episodes, 2000 frames
- LeRobotDataset variant: state32
- State: 32D TimeFeatureWrapper observation only
- Action: 6D padded action
- PandaStack-v1 real action dimensions: 0..3
- Padding dimensions: 4..5
- SmolVLA base model: lerobot/smolvla_base
- Training steps: 1000
- Batch size: 2

Validated:
- Training reached step 1000.
- Checkpoints were created at steps 250, 500, 750 and 1000.
- Offline checkpoint comparison was run against the PandaStack-v1 TQC teacher.
- Checkpoint 000750 was the best checkpoint in this run by L2 distance on real action dimensions 0..3.

Best offline checkpoint in this run:
- checkpoint: 000750
- base_l2_all_6d: 0.109474
- checkpoint_l2_all_6d: 0.100028
- improvement_all_6d: 0.009445
- checkpoint_l2_real_0_3: 0.100028
- improvement_real_0_3: 0.009445

Comparison note:
The 1000-step run shows a stronger relative improvement than the 500-step run. However, this is still an offline imitation metric against a weak official teacher and does not prove PandaStack-v1 rollout success.

Conclusion:
The 1000-step SmolVLA training run validates that the PandaStack-v1 state32 pipeline can train for a full pass over the 2000-frame dataset and improve offline imitation of the TQC teacher. Rollout success remains unproven.
