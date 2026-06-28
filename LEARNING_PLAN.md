# OpenVLA 学习计划 / Learning Roadmap (Checklist)

> 一份「实验驱动」的可续接学习清单。**这是进度的唯一真相来源 (single source of truth)。**
> 即使对话上下文被清空,只要读这份文件就能接着干。

---

## 🧭 如何使用 / How to resume (新会话先读这里)

1. 看下面的 **👉 RESUME HERE** 指针 —— 它指向「下一个该做的具体动作」。
2. 完成一项就把 `[ ]` 改成 `[x]`,并在底部 **进度日志** 记一行(日期 + 干了啥 + 卡在哪)。
3. 一个 Phase 全部勾完,就把 **RESUME HERE** 指针移到下一个 Phase 的第一项,并更新顶部状态表。
4. 规则:**勾选 = 已验证能跑/能讲清楚**,不是「大概做了」。

最后更新:2026-06-28

---

## 👉 RESUME HERE

> **下一步:Phase 3 —— 自己做 LoRA 微调(需 GPU)。详细 runbook 已写好:`experiments/04-lora-finetune/README.md`,下次开工照着敲。计划 RunPod 1× A100 80GB。Phase 2 已全部完成。**
> 📋 Phase 3 开工三件事:① `git clone hf.co:datasets/openvla/modified_libero_rlds`(~10GB,现成 RLDS,不用手动转);② `vla-scripts/finetune.py` 跑 LoRA r=32(先 `--max_steps 5000` 看 loss 下降);③ 复用 Exp 02 评测脚本(改 `--pretrained_checkpoint`)做三方对比。开工待定:步数预算 / batch 8+accum2 vs 16 / wandb 开关。
> ✅ Phase 2 完成:Exp 03 三个 notebook 全跑通 —— ①动作 tokenization(`action_tokenization.ipynb`)②视觉双编码器(`vision_encoders.ipynb`)③prompt 模板↔tokenizer(`prompt_template.ipynb`)。本地 `.venv` 已配齐(numpy/transformers/torch/timm,纯 CPU)。
> ✅ Phase 1 完成:Exp 02 官方 finetuned checkpoint 评测 = **62/73 = 84.9%**(libero_spatial),≈ 官方 84.7%,复现成功。README 已收尾。
> 待办(可选):把 RunPod 上的 `rollout.gif` 下载放进 `experiments/02-finetuned-evaluation/`(README 已引用)。
>
> **学习方式备注**:用「老师讲解 + 苏格拉底式提问 + 亲手跑小 notebook」推进,讲完一段出题验收。已啃透:动作=自回归 token 预测/256 bin/argmax/为什么恒 7 个/双视觉编码器(SigLIP 语义 + DINOv2 几何)。
> **Phase 3 GPU 选型**:RunPod `1× A100 80GB`(和 Exp 02 同 Ampere 架构,flash-attn==2.5.5 环境直接复用,免重配)。省钱备选 L40S/A6000 48GB。

### 🐛 已定位的根因:eager attention 导致 off-by-one,模型从不产出动作

- 真实报错:`attn_weights + causal_mask` 处 `RuntimeError: tensor a (287) must match tensor b (286) at dim 3`(Llama eager attention)。
- cell 13 显示**每一步都抛此异常并被 try/except 吞掉** → 30 episode 全 `success=False`,成功率 0.0(不是模型差,是根本没动作)。
- **根因 = cell 9 把 `flash_attention_2` 改成 `eager`**。OpenVLA 的 prismatic forward 拼接 256 图像 patch 后,4D causal mask 差 1;flash_attention_2 路径不物化/相加这个 mask(默认 causal),所以一直绕过;eager(及 sdpa)会显式相加 → 炸。
- **修复**:① `sed -i 's/attn_implementation="eager"/attn_implementation="flash_attention_2"/' .../openvla_utils.py` 撤销 cell 9;② `MAX_JOBS=4 pip install flash-attn==2.5.5 --no-build-isolation`(官方 pin,A100 编译 10-20 min);③ 先 `--num_trials_per_task 1` 验证不再抛异常。
- 收尾:cell 19 把官方 `Caught exception` 写 log 的逻辑删了,调试完恢复。

---

## 状态总览

| Phase | 主题 | 状态 |
| --- | --- | --- |
| 0 | 零样本复现（Exp 01） | ✅ Done |
| 1 | 跑通并读懂「评测」（Exp 02） | ✅ Done（84.9%) |
| 2 | 架构内部原理（读论文 + 拆代码） | ✅ Done（动作/视觉/prompt 三个 notebook 全跑通） |
| 3 | 自己做 LoRA finetune（Exp 03） | ⬜ 未开始 |
| 4 | 对比与消融 | ⬜ 未开始 |
| 5 | 拓展（新任务 / OFT） | ⬜ 未开始 |

---

## ✅ Phase 0 — 零样本复现（已完成，存档参考）

- [x] 加载 `openvla/openvla-7b`，`bfloat16` 跑通
- [x] 单步 `predict_action`，理解 7-DoF 动作向量
- [x] LIBERO 闭环 rollout，输出 `rollout.gif`
- [x] 踩坑归档:prompt 格式、`unnorm_key=bridge_orig`、headless 渲染、`pixel_values` dtype

---

## ✅ Phase 1 — 跑通并读懂「评测」（已完成,84.9%)

**做什么**
- [x] 选定 finetuned checkpoint = `openvla/openvla-7b-finetuned-libero-spatial`(官方,suite `libero_spatial`,`--center_crop True`)
- [x] 走官方评测脚本 `experiments/robot/libero/run_libero_eval.py`(不自己搭评测)
- [x] 解决环境坑链:eager→flash_attention_2 + flash-attn==2.5.5;protobuf<5 + tfds-metadata 1.14 + tfds 4.9.3;wandb==0.16.6
- [x] smoke test 跑通(每任务 1 条):9/10 = 90%,确认复现成功
- [x] 换 `MUJOCO_GL=egl` 提速后跑多条 rollout(seed 7),拿到 **62/73 = 84.9%**(跑到 73 ep 被中断,够用)
- [x] 弄清 `unnorm_key`:官方脚本从 checkpoint 自带 `dataset_statistics.json` 自动选,无需手填(对比 Exp 01 手填 `bridge_orig`)
- [x] 填满 Exp 02 README:checkpoint / task suite / eval 协议 / 对比表 / 坑链
- [x] 下载 `rollout.gif` 放进 Exp 02 文件夹(已修文件名空格 `rollout .gif`→`rollout.gif`)

**学完要能讲清楚（讲不清就别进 Phase 2）**
- [ ] 「成功率」怎么判定?(LIBERO 的 goal predicate)
- [ ] 为什么 finetuned 在 `libero_spatial` 上远好于零样本?(零样本相对 BridgeData 是 OOD)
- [ ] finetuned 的 `unnorm_key` 为什么和零样本不同?

**参考**:OpenVLA repo `experiments/robot/libero/`;LIBERO 4 个 suite(spatial/object/goal/long)

---

## ⬜ Phase 2 — 架构内部原理（从「会用」到「懂」的分水岭）

> 建议新建 `experiments/03-architecture-deepdive/`,写一个**纯分析 notebook**(不训练,只拆解)。
> ⚠️ 故意排在「自己训练」之前 —— 否则训练时模型对你是黑箱。

**做什么**
- [x] 打印 OpenVLA 的**动作 tokenization**:7-DoF 连续动作 → 256 bin → 映射到 Llama 词表中 256 个最少用 token(`action_tokenization.ipynb`)
- [x] 亲手把一个动作向量 encode 成 token、再 decode 回来,验证往返一致(官方 ActionTokenizer 对账一致,token 落 31744–31999)
- [x] 确认**视觉编码器** = SigLIP + DINOv2 双编码器特征拼接;看 patch 数、特征维度(`vision_encoders.ipynb`:256 patch × (1024+1152)=2176)
- [x] 确认 **backbone** = Llama-2 7B,动作预测本质是**自回归 token 预测**(不是回归头)
- [x] 把 prompt 模板 `In: What action should the robot take to {instruction}?\nOut:` 和 tokenizer 对应关系理清(`prompt_template.ipynb`:20 个文字 token,小 id 区,与动作尾部 31744–31999 互不重叠;`Out:` 后接 7 个动作 token)

**学完要能讲清楚**
- [x] 为什么能用「LM 生成 token」输出连续机器人动作?(离散化 + 复用低频 token + 自回归协调)
- [x] 256 bin 怎么切?(归一化到 [-1,1] 后 `np.linspace` 均匀分箱;decode 取格子中点)
- [x] 为什么要双视觉编码器?(SigLIP 语义「是什么」 + DINOv2 空间几何「在哪/怎么抓」)
- [x] 附带:argmax=取位置;生成恒 7 个 token(数量由代码 `range(7)` 定、内容由微调定);compounding error

**参考**:OpenVLA 论文(architecture / action tokenization 节);Prismatic VLM;代码 `prismatic/`

---

## ⬜ Phase 3 — 自己做 LoRA Finetuning（= Exp 04）

> 📋 **完整 runbook(可照敲命令 + 开工决策)见 `experiments/04-lora-finetune/README.md`。** 下面是要点。

**做什么**
- [ ] 准备数据:**直接下载官方现成 RLDS**(`git clone hf.co:datasets/openvla/modified_libero_rlds`,~10GB,含 spatial/object/goal/10),不用手动转
- [ ] 配 LoRA:官方默认 `r=32` / `dropout=0` / `lr=5e-4`;记下可训练参数占比(看 finetune.py 启动 log)
- [ ] 跑训练(`torchrun ... vla-scripts/finetune.py`),记录:显存、单 step 耗时、总时长、loss 曲线;先 `--max_steps 5000` 验证能学,再考虑长跑
- [ ] 用**自己训的 checkpoint** 跑 **Exp 02 同一套评测**(改 `--pretrained_checkpoint`)→ 零样本 / 官方 finetune / 自己 LoRA 三方对比
- [ ] 注意:LoRA 每次 save 会 `merge_and_unload` 合并进 base 并写出 `dataset_statistics.json`(评测反归一化要用)

**学完要能讲清楚**
- [ ] LoRA 改了哪些层?为什么只训这些就够?
- [ ] 训练 loss 是什么?(动作 token 的 cross-entropy)
- [ ] 为什么必须保存 `dataset_statistics.json`?(评测时 `unnorm_key` 靠它反归一化)

**参考**:OpenVLA repo `vla-scripts/finetune.py`;LoRA 原论文

---

## ⬜ Phase 4 — 对比与消融（挑 1–2 个深入,别贪多）

- [ ] **LoRA vs 全量 finetune**:成功率 / 训练成本 / 显存 三方权衡
- [ ] **量化推理**:4-bit / 8-bit,看显存↓ 和 成功率/延迟 的代价
- [ ] **action chunking / 推理频率**:每次预测几步、控制频率对 rollout 的影响
- [ ] 学完能讲清楚:成功率—训练成本—推理显存 三角里,各方案甜点区在哪

---

## ⬜ Phase 5 — 拓展（远期,选做）

- [ ] 换**新任务 / 新机器人本体**做 finetune,体会泛化与数据需求
- [ ] 跟进 **OpenVLA-OFT**(并行解码、连续动作头、action chunking),和 Phase 2 的「自回归离散 token」做对照

---

## 📊 总对比表（整个 repo 的成果证据,边做边填）

| 方案 | task suite | success rate | 训练成本 | 推理显存 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 零样本(Exp 01) | libero_spatial | _TODO_ | — | ~14GB | 相对 BridgeData OOD |
| 官方 finetune(Exp 02) | libero_spatial | **84.9%**(62/73) | — | ~14GB | ckpt `openvla-7b-finetuned-libero-spatial`;≈ 官方 84.7% |
| 自己 LoRA(Exp 03) | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |

> 规则:所有行用**同一套评测协议**(同任务、同 seed、同成功判据),否则数字之间没法比。

---

## 📝 进度日志 / Progress Log

> 格式:`YYYY-MM-DD — 做了啥 / 卡在哪 / 下一步`

- 2026-06-13 — 建立本计划文档。现状:Exp 01 完成;Exp 02 在调 `flash_attn`→`eager`、图像 resize、traceback。下一步见 RESUME HERE。
- 2026-06-13 — 从 notebook 确认 Exp 02 = 官方 ckpt `openvla-7b-finetuned-libero-spatial` + 官方 `run_libero_eval.py`。
- 2026-06-13 — 定位根因:cell 9 的 `eager` 改动导致 `causal_mask` off-by-one(287 vs 286),每步异常被吞、成功率 0。修复=回退 flash_attention_2 + 装 flash-attn==2.5.5。
- 2026-06-13 — 重建 notebook(23→11 cell,删调试 cell + eager bug,flash-attn 挪到评测前)。RunPod 配置见对话。
- 2026-06-13 — 新拦路点:import 阶段 `ImportError: cannot import name 'runtime_version' from 'google.protobuf'`(tfds 的 pb2 需 protobuf≥5.26)。修复:TF≥2.16 → `pip install -U "protobuf>=5.26.1"`;TF=2.15 → `pip install "protobuf<5" "tensorflow-metadata<1.15"`。先诊断 TF/protobuf 版本再定。→ 走降级:`pip install "protobuf<5" "tensorflow-metadata==1.14.0" "tensorflow-datasets==4.9.3"`,过了。
- 2026-06-13 — 连锁错:`run_libero_eval.py` 无条件 `import wandb`,wandb 的 pb2 与 protobuf 4.x 不匹配(`cannot import name 'Imports'`)。修复:`pip install "wandb==0.16.6"`(兼容 protobuf<5)。
- 2026-06-13 — ✅ 评测跑通!smoke test(每任务 1 条)9/10 = 90%,与官方 ~85% 吻合,确认复现成功。osmesa 慢(3.3min/ep)。下一步:换 egl,跑每任务 ≥10 条拿稳定数字。
- 2026-06-13 — 正式跑(10/task,seed 7)中途 kernel 断,跑到 73 ep = **84.9%**,≈ 官方 84.7%。教训:长跑批要在 JupyterLab 终端用 `nohup ... &` 脱离 kernel,`tail -f` 看 log。
- 2026-06-13 — ✅ **Phase 1 完成**。决定不重跑(73 ep 够),84.9% 定为 Exp 02 结果。Exp 02 README 已写完(setup + 对比表 + 坑链 + takeaways)。待办:下载 rollout.gif。下一步:Phase 2 或 3。
- 2026-06-28 — 开 Phase 2。建 `experiments/03-architecture-deepdive/` + 本地 `.venv`(纯 CPU,不用 GPU)。① 动作 tokenization notebook 跑通:手写 256-bin 逻辑与官方 ActionTokenizer 对账一致,token 落 31744–31999,量化误差 < 半格宽。② 视觉双编码器 notebook 跑通(timm 加载 DINOv2 ViT-L/14 + SigLIP SO400M/14 @224):各 256 patch,DINOv2 1024-d(带 5 个 cls/register 前缀,切掉)+ SigLIP 1152-d → 拼成 256×2176。教训:DINOv2 默认 518px 要强制 224。下一步:prompt 模板 ↔ tokenizer,收尾 Phase 2。
- 2026-06-28 — ✅ **Phase 2 完成**。③ prompt 模板 notebook 跑通:`In: What action...? \nOut:` + 指令 = 20 个文字 token(含 `<s>` BOS),全是小 id(常用区),与动作 token 尾部 31744–31999 互不重叠;`bowl`→`bow`+`l` 子词切分;完整输入 = 256 视觉 token + 20 文字 token = 276,读完从 `Out:` 后吐 7 个动作 token。整条「图+文→动作」链路打通。下一步:Phase 3 LoRA(RunPod A100 80GB)。
