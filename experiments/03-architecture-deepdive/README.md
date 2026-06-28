# Exp 03 · Architecture Deep-Dive

> **Pure analysis. No training, no GPU.** This is the watershed between "knowing how to use it" and "understanding it" — turning the model from a black box into a white box *before* fine-tuning it myself.
> Corresponds to `LEARNING_PLAN.md` Phase 2.

## Progress

| Topic | notebook | status |
| --- | --- | --- |
| (1) Action tokenization round-trip | `action_tokenization.ipynb` | ✅ |
| (2) Vision encoders: SigLIP + DINOv2 | _TODO_ | ⬜ |
| (3) Prompt template ↔ tokenizer mapping | _TODO_ | ⬜ |

## (1) Action Tokenization — Key Takeaways

OpenVLA has **no regression head**. A 7-DoF continuous action becomes a "token" like this:

```
continuous value (normalized to [-1,1])  →  np.digitize into 1 of 256 bins  →  token_id = vocab_size − bin
```

- **256 bins** = `np.linspace(-1, 1, 256)`, uniform split; each bin ≈ 0.0078 wide.
- **Reuse the tail of the vocab**: token_ids land in `31744 ~ 31999` (= `32000-256` to `32000-1`), all rarely-used tokens, so they don't disturb the model's language ability.
- **Quantization error**: decode picks the bin center, so error is always < half a bin width (≈ 0.0039). This is the inherent cost of discretization.
- **gripper (a binary dim)**: the mechanism is generic — it still gets squeezed into 256 bins and in practice only occupies the two end bins. Slightly wasteful, but harmless.
- The normalization statistics come from the checkpoint's bundled `dataset_statistics.json` — i.e. what `unnorm_key` points to back in Phase 1.

### How to run

```bash
# From the repo root, set up an isolated venv (one-time):
python3 -m venv .venv
.venv/bin/python -m pip install numpy transformers sentencepiece jupyterlab ipykernel
.venv/bin/python -m ipykernel install --user --name openvla-study --display-name "Python (openVLA-study)"

# Open the notebook and select the "Python (openVLA-study)" kernel:
.venv/bin/jupyter lab action_tokenization.ipynb   # or open the .ipynb in VS Code
```

- **Step 1** (pure numpy, zero downloads) runs instantly on any machine with numpy.
- **Step 2** downloads a few-MB Llama-2 tokenizer (`NousResearch/Llama-2-7b-hf`, no HF login required) and reconciles against the official logic, verifying the hand-written result matches. Still **no GPU, no 7B weights**.
