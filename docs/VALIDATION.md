# Validation record 2026 10 08

This record distinguishes executed local checks from future GPU qualification.

| Check | Actual result | What it does not establish |
|---|---|---|
| Python 3.10.20 unit suite on macOS arm64 | 17 passed | No GPU training, no model quality or speed evidence |
| Offline wheel build without installing dependencies | Succeeded | Not a validated training environment or published package |
| Real existing Qwen2.5-0.5B-Instruct tokenizer and local metadata preflight | Three SFT fixture rows tokenized, maximum 63 tokens; no model weights loaded | Not a forward/backward/checkpoint test |
| Local training readiness | Correctly blocked: macOS, no visible CUDA GPUs, missing pinned training backend | Not an instruction to install CUDA on macOS |
| Runner success/failure boundary | Mocked subprocess, logs and nonzero status tested | Not a real training run |

No software environment was installed, no model weights were downloaded, no training job was submitted and no cloud GPU was rented for these checks. The built wheel remains a local build artifact.

## Required GPU promotion gates

Record independently for each method/model/backend configuration: immutable model revision, software lock and image, driver/device, dataset hashes and held-out split, actual argv and saved backend arguments, role allocation, first/peak memory, finite loss/reward/gradient metrics, wall time and cost, saved checkpoint reload, and baseline-versus-trained held-out evaluation.

GRPO additionally requires nonzero reward variance, reward parser verification and reward-hacking examples. PPO requires a calibrated compatible scalar reward checkpoint, value-loss/KL traces and confirmation of intended rollout/update cycle count. Failed runs remain evidence and must not be excluded from the support record.
