# Experiment 02 — Finetuned Checkpoint Evaluation

> 🚧 **Work in progress** — running this experiment now.

Evaluating a **finetuned** OpenVLA checkpoint (not self-trained yet) and comparing its closed-loop performance against the zero-shot baseline from [Experiment 01](../01-zero-shot-reproduction/).

> Goal: measure how much a task-adapted checkpoint improves over the out-of-the-box weights on the same LIBERO task, using an identical evaluation setup.

## Setup

<!-- TODO: fill in after the run -->

- **Checkpoint**: _TODO — which finetuned checkpoint (name / HF repo / source)_
- **Task suite**: _TODO — e.g. `libero_spatial`, task index, same as Exp 01 for a fair comparison_
- **`unnorm_key`**: _TODO — which normalization stats this checkpoint expects_
- **Eval protocol**: _TODO — number of rollouts, max steps, seed(s), success criterion_

## Result

<!-- TODO: add rollout.gif and key numbers once available -->

_TODO: rollout GIF + success rate vs. baseline._

## Comparison vs. Baseline (Exp 01)

| Metric | Exp 01 (zero-shot) | Exp 02 (finetuned) |
| --- | --- | --- |
| Success rate | _TODO_ | _TODO_ |
| Notes | OOD w.r.t. BridgeData | _TODO_ |

## Notes & Gotchas

<!-- TODO: record anything that tripped you up this run -->

- _TODO_

## Takeaways

<!-- TODO: what did this tell you, and what's next? -->

- _TODO_ — likely motivates [Experiment 03 (LoRA finetuning)](../) as the next step.
