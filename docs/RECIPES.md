# Pinned backend contracts and parameter meanings

The first adapter emits ms-swift **4.5.3** commands with **TRL 0.26.0**. SFT invokes `python -m swift.cli.sft`; DPO/RM/PPO/GRPO invoke `python -m swift.cli.rlhf --rlhf_type METHOD`. Multi-GPU uses PyTorch `torch.distributed.run --standalone` on one node. Backend defaults not set by the plan must be captured from the backend's saved arguments; a command recipe is not a complete reproducibility lock.

## Data contracts

All rows are UTF-8 JSONL with `messages`, an optional leading system turn and alternating user/assistant text turns. No image/audio/video/tool-call data is accepted by this adapter. Unknown fields are rejected to expose format mistakes.

| Method | Last message | Additional fields | Learning signal |
|---|---|---|---|
| SFT | assistant | none | Supervised answer tokens; upstream masking/template applies |
| DPO | assistant, chosen answer | `rejected_response` | Same-prompt preference relative to reference |
| RM | assistant, chosen answer | `rejected_response` | Scalar reward ranking |
| GRPO | user | `solution` | Online sampled groups, verified mathematical accuracy |
| PPO | user | none | Online sampled answers scored by a separate scalar reward model |

Train/eval prompt hashes must not overlap. This is exact normalized-JSON matching, not semantic deduplication. Doctor uses the checkpoint tokenizer to check prompt/completion token budgets; it does not assert that AutoTokenizer and every ms-swift mask are identical. Inspect backend encoded examples in the first GPU qualification. The current tool does not scan every pretrained tensor for integrity or load the model during planning.

## Parameters visible in each plan

| Spec or emitted option | Meaning and default |
|---|---|
| `method` | Loss/workflow: sft, dpo, rm, grpo, ppo; not automatically interchangeable |
| `tuner` → `tuner_type` | `lora` freezes base weights; `full` trains the model parameters |
| `model`, `train`, `eval`, `python`, `output` | Local checkpoint, JSONL paths, trusted backend executable, fresh experiment directory |
| `gpus` | 1–8 visible GPUs on one Linux host; selects first N of the existing GPU mask; no remote provisioning |
| `max_length` | Token budget (default 2048); bounded by config context and checked against data; not a training curriculum |
| `steps` → `max_steps` | Optimizer-step smoke budget, default 20 except PPO (see cycle mapping below); not promised convergence |
| `batch` | Per-device train/eval microbatch, default 1 |
| `accumulation` | Microsteps per optimizer update, default 16, 4 for GRPO, 2 for PPO |
| `lr` | Learning rate, default 1e-4 for LoRA SFT/RM and 1e-5 otherwise; tune against task/model and held-out quality |
| `rank` | LoRA rank, default 8; alpha = 2 × rank, target modules = all-linear; no effect in full mode |
| `torch_dtype` | BF16, fixed in first adapter; does not mean all optimizer tensors use BF16 |
| `gradient_checkpointing` | true; trade recomputation for saved activation memory |
| `deepspeed` | Absent on one GPU; zero2 for multi-GPU LoRA, zero3 for full; conservative starting recipe, not searched optimum |
| `seed` | 42; not a guarantee of bitwise distributed reproducibility |
| `split_dataset_ratio` | 0; use explicit independent `val_dataset`, never silently split supplied training data |
| `logging_steps` | 1, expose backend optimizer-step output |
| `eval_steps`, `save_steps` | min(10, steps); backend eval/save policy still applies; PPO's backend evaluation semantics differ from SFT loss evaluation |
| `save_total_limit` | 2 checkpoints retained by backend; monitor actual disk space |
| `report_to` | none; no automatic external experiment-tracking upload |
| `dataloader_num_workers` | 0 for first-run portability; tune loading throughput later |
| DPO `beta` | 0.1, controls the preference objective's reference-relative scale |
| GRPO `generations` | Group size, default 4 and >=2; GPUs × batch × accumulation must be divisible by group size |
| GRPO `num_generations_eval` | 1 for the initial small evaluation batch; reports single-sample reward, not pass@G |
| GRPO `reward_funcs` | accuracy only; requires math-verify; no universal or learned reward claim |
| GRPO `use_vllm` | false in first adapter; in-process Transformers generation |
| GRPO `beta`, `temperature` | 0.04 reference penalty and 0.9 sampling; finite reward variance must be inspected |
| `completion_length` | Online generation cap, default 256, smaller than max_length |
| PPO `reward_model` | Local complete scalar checkpoint, same tokenizer/template as policy; also initializes separate trainable value model upstream |
| PPO `temperature` | 0.7 for sampled rollouts |
| PPO `local_rollout_forward_batch_size` | 1 to limit initial rollout microbatch pressure; not a complete VRAM bound |

PPO's upstream defaults include four PPO epochs, KL coefficient .05, clipping .2, value coefficient .1, gamma 1 and lambda .95. These are recorded backend defaults, not values optimized by TessWeave. Before a serious PPO run, inspect the resulting backend arguments, KL/reward/value-loss curves and sampling behavior.

PPO in pinned TRL sizes training from episodes, not SFT's `max_steps`. The recipe removes max_steps and sets `num_train_epochs = (steps × GPUs × batch × accumulation + 0.5) / train_rows`. Upstream integer conversion yields the intended episode count; dividing by the global rollout batch gives `steps` rollout/update cycles. The half-episode offset protects against floating-point rounding below the intended integer. Each cycle contains internal PPO epochs, not a single optimizer update. Require enough rows for a full global train and eval batch before launching. [Pinned PPO batching and episode loop](https://github.com/huggingface/trl/blob/v0.26.0/trl/experimental/ppo/ppo_trainer.py).

For SFT/DPO a nominal update batch is GPUs × microbatch × accumulation sequences. For DPO each example also contains two responses; tokens/compute are not simply the SFT batch. GRPO repeats prompts across sampled generations and PPO has rollout/update minibatches; do not reuse that simple SFT formula as their effective prompt/token batch.

## Memory roles rather than a false universal minimum

With LoRA, the frozen policy still occupies device memory. Trainable adapter gradients and optimizer moments are smaller, but activations remain. DPO adds chosen/rejected forward work and reference evaluation. LoRA reference evaluation may use the policy with adapters disabled; full-parameter training needs a separate frozen reference. GRPO adds generation KV, sampled sequences and reference work but no value model. PPO adds frozen reward and separate trainable value models plus rollout storage. Shared tokenizers do not imply shared weight allocation.

Single-GPU fit is **not calculated** by this first adapter. Neither file size nor total parameters alone establishes peak memory. Studio's resource planner is a separate system and is not silently being reused here. Auto topology, measured calibration, QLoRA, reference offload and optimized rollout placement require later adapters and recorded tests.

## Upstream source traceability

Reviewed against the local ms-swift 4.5.3 source distribution. Upstream paths for inspection:

- [LoRA SFT example](https://github.com/modelscope/ms-swift/blob/v4.5.3/examples/train/lora_sft.sh)
- [DPO LoRA example](https://github.com/modelscope/ms-swift/blob/v4.5.3/examples/train/rlhf/dpo/lora.sh)
- [PPO LoRA example](https://github.com/modelscope/ms-swift/blob/v4.5.3/examples/train/rlhf/ppo/lora.sh)
- [Transformers GRPO example](https://github.com/modelscope/ms-swift/blob/v4.5.3/examples/train/grpo/internal/transformers.sh)
- [RLHF arguments and defaults](https://github.com/modelscope/ms-swift/blob/v4.5.3/swift/arguments/rlhf_args.py)
- [Role construction and value initialization](https://github.com/modelscope/ms-swift/blob/v4.5.3/swift/pipelines/train/rlhf.py)
- [Math accuracy and format rewards](https://github.com/modelscope/ms-swift/blob/v4.5.3/swift/rewards/orm.py)

The upstream examples are evidence for command/role semantics, not measured evidence for our recipes. Original JSONL examples and workflow code in this repository are not copied training datasets or vendored upstream trainer implementations.
