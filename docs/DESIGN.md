# Guided post training design

The product goal is an actionable training assistant: explain what supervision is missing, propose the smallest useful experiment, expose the command and evidence, then execute only after approval. A UI that merely exposes hundreds of backend parameters does not satisfy that goal.

## Implemented first vertical slice

`guide → local model and data checks → versioned plan → trusted environment probe → explicit approval → real backend process → audit artifacts`.

The workflow layer owns decisions, contracts, input validation and execution evidence. ms-swift owns the trainer, gradients, optimizer, checkpoint serialization and RL rollout mechanics. A future backend adapter can replace ms-swift without changing what the user must prove about data, reward and quality.

The CLI and plans are English by default; Chinese walkthroughs explain the same flow. Full bilingual UI strings and guided Studio forms are future work. No external service or account is required for the workflow layer.

## Next development gates

1. **Real algorithm qualification.** On a small text checkpoint run SFT LoRA/full, DPO, math GRPO and PPO with a verified compatible reward checkpoint. Record package lock, model revision, loss/reward traces, checkpoint reload, evaluation and actual GPU peaks. Do not promote a mocked test to a passed training run.
2. **A guided Studio workflow.** Ask goal, model, supervision and budget first. Present missing requirements with data examples. Show proposed algorithm, adaptation method, all memory roles, commands and decision evidence. Installing environments or submitting cloud work requires a separate explicit approval.
3. **Close the artifact loop.** Discover checkpoints, retain base-model identity for adapters, evaluate against unchanged baselines, review merge/export recipes, and register validated serving artifacts. Auto resume needs checkpoint/input compatibility validation; a repeated command is not safe recovery.
4. **Calibrated resource selection.** Integrate the existing Studio planner through a stable API, then calibrate activations/rollout/KV/communication with small pilot runs. Search parallelism, ZeRO and role placement with actual hardware and task constraints. Explain why a cheaper option is rejected. Current fixed conservative single-node recipes are not this optimizer.
5. **Fast RL and broader models.** Add separately validated vLLM rollout adapters, then investigate TessWeave Engine rollout. Require log-probability/sampling/template/weight-update consistency and policy-lag controls; an OpenAI-compatible text endpoint alone is insufficient. Extend to multimodal and MoE only with distinct memory/data and correctness contracts.

## Where originality can come from

The research opportunity is an evidence-driven training decision layer: selecting the shortest effective experiment, diagnosing zero-variance or exploit-prone rewards, matching rollout/trainer placement to memory and communication limits, and learning cost/quality predictions from measured pilot runs. None of these are claimed as completed algorithms in the preview.

Similarly, Engine should optimize a latency-throughput frontier rather than one concurrency point. Dynamic prefill and verification budgets, cache pressure and queue age interact; changes must be evaluated across load levels and include fairness and failure outcomes. Train, Engine and Studio share experiment contracts, not a claim of universal superiority.

## Safety and reproducibility boundaries

Only trusted Python environments and model/data directories should be used. Plan building does not import model code. Backend probing does execute the chosen Python and installed libraries. Approval hashes detect accidental plan/input changes, not malicious local users. Weight size/mtime inventories are not full cryptographic integrity checks. Backend offline environment variables are not a network firewall.

The runner records selected environment overrides but never dumps every environment variable, which could expose credentials. Backend stdout may still include sensitive data; logs must be reviewed before publication. Do not store credentials in specification files. No shell is used for training subprocess invocation.
