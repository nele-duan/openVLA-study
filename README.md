# OpenVLA Study

A hands-on study of [OpenVLA-7B](https://github.com/openvla/openvla), a 7B Vision-Language-Action (VLA) model for robot control. Each experiment is a self-contained folder documenting one step of the journey — from reproducing the released checkpoint to finetuning and beyond.

> OpenVLA takes a robot camera view and a natural-language instruction, and predicts the robot's next 7-DoF action.

## Experiments

| # | Experiment | Status | Description |
| --- | --- | --- | --- |
| 01 | [Zero-Shot Reproduction](experiments/01-zero-shot-reproduction/) | ✅ Done | Run the **pre-finetuning** OpenVLA-7B checkpoint zero-shot: single-step prediction + closed-loop rollout in LIBERO |
| 02 | _Finetuning on LIBERO_ | 🔜 Planned | Finetune on LIBERO and compare rollout success against the zero-shot baseline |

Each experiment folder has its own README with the full pipeline, requirements, and notes.

## Preview

![rollout](experiments/01-zero-shot-reproduction/rollout.gif)

_Experiment 01: zero-shot closed-loop rollout on a LIBERO `libero_spatial` task._

## Repository Layout

```
openVLA-study/
├── README.md                              ← you are here (overview + index)
└── experiments/
    └── 01-zero-shot-reproduction/         ← pre-finetuning baseline
        ├── README.md
        ├── openVLA-reproduce.ipynb
        ├── first_frame.png
        └── rollout.gif
```

## References

- OpenVLA: https://github.com/openvla/openvla
- Model weights: https://huggingface.co/openvla/openvla-7b
- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO

---

A personal learning project — not an official implementation.
