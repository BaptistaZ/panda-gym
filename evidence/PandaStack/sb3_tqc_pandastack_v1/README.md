# SB3 TQC PandaStack-v1 Teacher Evaluation

This folder stores compact evidence from the evaluation of the official Hugging Face model `sb3/tqc-PandaStack-v1`.

The model was tested in an isolated legacy environment using:
- Python 3.8
- gym 0.21.0
- panda-gym 1.1.1
- stable-baselines3 1.8.0
- sb3-contrib 1.8.0
- PandaStack-v1
- official TimeFeatureWrapper

The official model loaded successfully and produced valid 4D actions for PandaStack-v1.

However, the local 20-episode evaluation showed weak teacher performance:
- success_rate: 0.10
- mean_initial_distance: 0.2508
- mean_best_distance: 0.2493
- mean_final_distance: 0.2544
- mean_improvement: 0.00148
- mean_steps: 90.1

The official Hugging Face `results.json` also reports a weak deterministic evaluation:
- mean_reward: -99.9
- n_eval_episodes: 10

Conclusion: the model is technically executable in the correct PandaStack-v1 environment, but it is not strong enough to be used directly as a reliable teacher for generating LeRobot demonstrations for SmolVLA.
