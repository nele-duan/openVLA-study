# Exp 04 · My Own LoRA Finetune (Phase 3)

> **Status: PLANNED — runbook for next session, not run yet.**
> Goal: finetune `openvla/openvla-7b` myself with **LoRA** on **libero_spatial**, then eval with the *exact same harness as Exp 02*, to get a three-way comparison:
> **zero-shot (Exp 01) vs official finetune (Exp 02) vs my LoRA (this)**.
>
> This is the first time we actually **train**. Needs a GPU. Plan: RunPod **1× A100 80GB** (same Ampere arch as Exp 02 → flash-attn env reuses cleanly).

---

## 0. Why this design

- **Reuse Exp 02's whole environment.** Same box, same `flash-attn==2.5.5`, same protobuf/wandb pins. Don't re-fight that chain.
- **Same suite as Exp 02 (`libero_spatial`)** so the eval number is directly comparable (same task, same success predicate).
- **Pedagogical first, SOTA second.** First run is about *seeing the machinery* — trainable-param %, loss curve going down, the LoRA→merge→`dataset_statistics.json` flow. Matching 84.9% needs many steps/hours; a short run that clearly learns is the win for Phase 3.

---

## 1. Data — download pre-converted LIBERO RLDS (no manual conversion!)

The OpenVLA team already converted LIBERO demos to RLDS. **We do NOT hand-convert raw demos.**

```bash
# ~10 GB: libero_spatial / object / goal / 10, all in RLDS
git clone git@hf.co:datasets/openvla/modified_libero_rlds
# (or the https form if no SSH key on the box)
```

➡️ **TODO on the box**: `ls modified_libero_rlds/` to confirm the exact `--dataset_name`
(it is a `*_no_noops` variant, e.g. `libero_spatial_no_noops` — verify the precise folder name before training).

---

## 2. Environment (on top of Exp 02's working env)

```bash
# Exp 02 already gives us: torch, transformers, flash-attn==2.5.5, protobuf<5, wandb==0.16.6, LIBERO sim
pip install peft                      # LoRA
pip install -r experiments/robot/libero/libero_requirements.txt   # if not already from Exp 02
```

Sanity: `python -c "import peft, flash_attn; print('ok')"`.

---

## 3. Train — `vla-scripts/finetune.py` (LoRA, single GPU)

Official defaults (from `FinetuneConfig`): `use_lora=True`, `lora_rank=32`, `lora_dropout=0.0`,
`batch_size=16`, `learning_rate=5e-4`, `grad_accumulation_steps=1`, `image_aug=True`,
`max_steps=200_000`, `save_steps=5000`. **We override max_steps/save_steps down for a first run.**

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir <DIR CONTAINING modified_libero_rlds> \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir runs \
  --adapter_tmp_dir adapter-tmp \
  --use_lora True --lora_rank 32 --lora_dropout 0.0 \
  --batch_size 8 --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --image_aug True \
  --max_steps 5000 \
  --save_steps 1000 \
  --wandb_project <my-project> --wandb_entity <my-entity>
```

Notes / decisions:
- **VRAM**: official reports ~72 GB at `batch_size=16` on one GPU. On A100-80GB that *fits* but is tight → start at **`batch_size 8 + grad_accum 2`** (same effective 16) for headroom; bump to 16 if memory is comfortable.
- **max_steps**: 5000 is a "does it learn?" run (watch loss drop, eyeball a few rollouts). A serious run toward ~85% is longer (plan a second, longer launch with `nohup` once the short one is clean). **Run long jobs detached** (`nohup ... &` + `tail -f`) — Exp 02 lesson: kernel disconnect kills the run.
- **wandb**: set your own project/entity (defaults point at the authors'). Or disable if you prefer.
- At each save: LoRA adapter is **merged into the base** via `merge_and_unload()` and the full merged model is written to `run_dir`, **with `dataset_statistics.json`** — that file is what eval's `unnorm_key` needs.

---

## 4. Eval — reuse the Exp 02 harness, just repoint the checkpoint

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint <runs/...my merged checkpoint dir> \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 10
# MUJOCO_GL=egl for speed (Exp 02 lesson)
```

Same protocol as Exp 02 (same suite, same seed, same success predicate) → the number is comparable.

---

## 5. Fill these in when done (evidence table)

| approach | suite | success rate | train cost | infer VRAM | notes |
| --- | --- | --- | --- | --- | --- |
| zero-shot (Exp 01) | libero_spatial | _TODO_ | — | ~14GB | OOD vs BridgeData |
| official finetune (Exp 02) | libero_spatial | **84.9%** | — | ~14GB | ckpt `openvla-7b-finetuned-libero-spatial` |
| **my LoRA (Exp 04)** | libero_spatial | _TODO_ | _TODO_ (steps×s/step) | ~14GB | r=32; _my run id_ |

---

## 6. "Be able to explain" checklist (Phase 3 graduation)

- [ ] Which layers does LoRA touch, and why training just those is enough? (low-rank adapters on attention/MLP projections; base frozen)
- [ ] What is the training loss? (cross-entropy on the **action tokens** — ties straight back to Exp 03)
- [ ] Why must `dataset_statistics.json` be saved? (eval reverses normalization via `unnorm_key` — ties back to Exp 03's [-1,1] normalization)
- [ ] Trainable-param % under LoRA r=32 (read it off the finetune.py startup log)

---

## 7. Scheduling, cost & the overnight pattern

**Planned start: Friday/Saturday afternoon.** Afternoon start is ideal — the risky setup happens
while awake/in daylight, and the long unattended training rolls into the evening/overnight.

### Cost estimate (RunPod A100-80GB, ~$1.7/hr; community/spot ~half)

| stage | rough time | rough cost |
| --- | --- | --- |
| env + data download (~10GB) + smoke test | 1–2 hr | ~$2–4 |
| short "does-it-learn" run (5k steps) | 3–5 hr | ~$6–10 |
| eval (Exp 02 harness, egl, ~100 episodes) | 1–2 hr | ~$2–4 |
| **subtotal: full pipeline + my own number** | **~6–9 hr** | **≈ $15–25** |
| optional long run toward ~85% (~20–30k steps) | 15–25 hr | ~$30–50 |

Budget **~$20–30** for a comfortable Phase 3. The single biggest unknown is **seconds/step** —
measure it in the first ~100 steps and extrapolate (don't trust the estimate above blindly).

### The overnight pattern (do NOT launch-and-sleep)

The first 1–2 hr (env, flash-attn, dataset name, OOM) is where things break — do it **awake**.

```
🌅 Afternoon (awake, ~1.5 hr):
   1. env + download data (~10GB)
   2. smoke test: run a few dozen steps, confirm loss prints, no crash, VRAM ok
   3. only once it's clearly running → launch the long run with nohup, then leave it
      (watch until loss is actually dropping before walking away)

🌙 Evening/overnight (unattended): training runs

🌅 Next morning (awake, ~1–2 hr): training done → run eval → fill the table in §5
```

Two money-savers:
- **Size `--max_steps` to your unattended window.** ~3 s/step → ~10k steps in ~8 hr. Set
  `--max_steps 10000` so it finishes around when you return (5k finishes in ~4 hr — fine for an afternoon).
- **Auto-stop the pod when done** so it doesn't idle-bill while you sleep. Append a stop command
  after training (e.g. `runpodctl stop pod $RUNPOD_POD_ID` or `poweroff`). A pod idling 5 hr ≈ wasted ~$8.

**First run: don't chase 85%.** Goal is a clean pipeline + loss visibly dropping + a few sane
rollouts. Even 60–70% (with "lower because fewer steps" understood) is a successful Phase 3.
Chase the full number in a separate, longer launch later.

---

## Open decisions to make at kickoff
1. **Step budget**: short "does-it-learn" 5k run first, or commit to a longer ~overnight run straight away?
2. **batch_size 8+accum2 vs 16**: depends on observed VRAM headroom on the A100.
3. **wandb on/off** and project name.
