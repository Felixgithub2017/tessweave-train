# TessWeave Train

**Guided post-training: tell the system your goal; receive a reviewable path from data to a real training run.**

Train is the post-training member of TessWeave. It separates the learning objective (SFT, DPO, PPO, GRPO, reward modeling) from the update strategy (LoRA or full parameters). It builds and executes local-checkpoint training commands through **ms-swift 4.5.3 / TRL 0.26.0**; it is not a new implementation of those optimizers.

**Engineering preview. No GPU training run is validated by this release.** The first adapter targets unquantized text Llama, Qwen2, Qwen3 and Mistral checkpoints on Linux/NVIDIA BF16 GPUs. Recognized architecture metadata is not a compatibility certification. Mac can run guidance, planning and unit tests; this adapter does not train on MPS. MoE, multimodal training, QLoRA, multi-node training, vLLM rollout and GUI integration are not implemented here yet.

[中文手把手指南](docs/GETTING_STARTED_ZH.md) · [Recipe contracts and parameters](docs/RECIPES.md) · [Design and development gates](docs/DESIGN.md) · [Actual validation record](docs/VALIDATION.md)

## Three products with separate responsibilities

| Product | Responsibility | Current relationship |
|---|---|---|
| [TessWeave Studio](https://github.com/Felixgithub2017/tessweave-studio) | Visual model engineering, hardware/resource planning, reviewed workflows and logs | Existing separate project; Train GUI integration is planned, not shipped |
| [TessWeave Train](https://github.com/Felixgithub2017/tessweave-train) | Guided objective selection, dataset contracts, pinned backend recipes, preflight, approved execution and audit records | This repository; CLI workflow preview |
| [TessWeave Engine](https://github.com/Felixgithub2017/tessweave-engine) | Independent inference runtime targeting individual-request latency and concurrent throughput | Not a PPO/GRPO rollout backend yet; Train currently uses upstream Transformers rollout |

Each can work independently. Train will integrate with Studio through a versioned plan/event contract, not by copying private UI state. An algorithm can graduate from command-contract support to validated training only after real runs, numerical/behavioral checks and recorded evidence.

## Start from your supervision

```bash
python -m pip install -e .
tessweave-train guide
tessweave-train wizard --out plan.json
```

| What you have | Initial route | What else is required |
|---|---|---|
| Correct input/output demonstrations | SFT + LoRA | Held-out demonstrations and task-quality evaluation |
| Chosen/rejected responses for identical prompts | DPO + LoRA | Instruction-capable starting model; preference-label audit |
| Math problems and machine-verifiable answers | GRPO + LoRA | Parseable solutions, several sampled responses per prompt, reward audit |
| Prompt set and a compatible scalar reward checkpoint | PPO + LoRA | Frozen reference/reward roles, separate trainable critic, rollout memory and reward validation |
| Preference pairs but no scalar reward model | RM, then a separately reviewed PPO experiment | Export a complete compatible reward checkpoint; a LoRA adapter alone is not that checkpoint |

These are starting hypotheses, not an assertion that RL is superior to SFT. “LoRA” changes trainable weights; it does not choose the loss or create missing labels/rewards. The small supplied JSONL files are **schema/smoke fixtures**, not useful quality-training corpora.

## First real backend workflow

On your intended Linux GPU host, prepare the reviewed backend environment described in the Chinese guide. No environment installation or model download is executed by this tool. `wizard` asks about supervision, model, data, environment and hardware, then creates the plan without requiring a hand-written spec. For repeatable scripted work, edit `examples/sft.spec.json` with actual local model, data and Python paths instead. Paths in the specification are relative to your current working directory unless absolute; the plan records absolute paths.

```bash
tessweave-train plan --spec examples/sft.spec.json --out plan.json
tessweave-train doctor --plan plan.json
# Inspect plan.json, its exact command, decisions and warnings first.
tessweave-train run --plan plan.json --approve THE_FULL_SHA256_PRINTED_BY_PLAN
```

Planning reads config/tokenizer metadata and a weight file inventory, never model code or tensors. It checks data shape and train/eval prompt overlap. Doctor runs the explicitly chosen trusted Python, checks Linux/CUDA/BF16/backend version, and tokenizes all data without loading weights. For GRPO it also checks math-answer parsing. Token lengths are an AutoTokenizer preflight, not a proof that every upstream training mask matches; inspect the first backend run.

Execution regenerates the plan and checks its approval hash before doing anything. Changed inputs require new review. It selects the first requested GPUs from `CUDA_VISIBLE_DEVICES`, preserving an existing device mask. It starts the real backend without shell interpolation and prints its output in the **current terminal**. Ctrl-C stops the child process group. It sets offline Hub options and requires local model/data paths; this is not an OS network sandbox.

## What is recorded

Each fresh run directory contains `plan.json`, `preflight.json`, `command.txt`, `console.log`, timestamped `events.jsonl`, `gpu.jsonl`, `status.json`, and backend artifacts under `artifacts/`. GPU samples are whole-device observations, not exclusive job attribution. Unavailable samples are explicitly marked. Trainer output is preserved verbatim; automatic loss-chart parsing and evaluation dashboards are not implemented.

Data hashes and tokenizer/config hashes detect changed inputs. Weight size/mtime inventory is **not** a cryptographic checkpoint digest. Use immutable, verified local snapshots. Logs and plans can contain dataset information; review before sharing, and do not commit run directories.

## Test and evidence boundary

```bash
python -m unittest discover -s tests -v
```

Tests exercise recipe construction, data leakage checks, role constraints, approval invalidation, environment blocking and runner logging. Metadata fixtures and mocked process boundaries are not training demonstrations. A real smoke run is still required for each algorithm/model/device/backend combination. No throughput, memory-fit, quality or “best training plan” claim is made by these tests.

MIT for project code. ms-swift and its dependencies retain their own licenses. See pinned upstream references in [recipe contracts](docs/RECIPES.md).
