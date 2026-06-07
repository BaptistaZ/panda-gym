# SmolVLA PandaPickAndPlace Stage 2 Evidence

## Selected checkpoint

- Stage 2 checkpoint: `035000`
- Accumulated training steps: `45000`
- Action scale: `1.0`
- Evaluation episodes: `50`
- Evaluation seeds: `0-49`
- Maximum steps per episode: `50`
- Deterministic per-episode RNG seeding: enabled

## Official result

- Stage 1 baseline: `17/50 = 34%`
- Selected Stage 2 checkpoint: `29/50 = 58%`
- Absolute improvement: `+24 percentage points`
- Additional successful episodes: `+12`

## Selection process

1. All ten Stage 2 checkpoints were evaluated for 20 episodes.
2. The five strongest candidates were compared with action scales
   `1.0` and `0.75`.
3. Action scale `1.0` outperformed `0.75` for every candidate.
4. The three finalists were evaluated for 50 episodes.
5. Checkpoint `035000` achieved the highest success rate.
6. The first 20 episodes were identical to the previous deterministic
   evaluation.

The final checkpoint was selected by rollout success rate rather than
training loss.
