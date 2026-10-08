"""Pinned ms-swift recipes. Planning never imports model code or downloads files."""
import hashlib
import json
import math
from pathlib import Path

BACKEND_VERSION = "4.5.3"
TRL_VERSION = "0.26.0"
METHODS = {"sft", "dpo", "grpo", "ppo", "rm"}
TEXT_TYPES = {"llama", "qwen2", "qwen3", "mistral"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def advise(goal):
    routes = {
        "examples": ("sft", "Prepare correct prompt/answer demonstrations; start with LoRA and a held-out set."),
        "preferences": ("dpo", "Prepare same-prompt chosen/rejected pairs. Start from an instruction-following checkpoint."),
        "math": ("grpo", "Prepare prompts plus verifiable mathematical solutions, NOT answers inside the prompts. Audit reward parsing first."),
        "reward-model": ("ppo", "Provide a scalar reward checkpoint with the SAME tokenizer/template as the policy. A trainable value model is also allocated."),
        "train-reward": ("rm", "Train a scalar reward model from chosen/rejected pairs before attempting PPO."),
    }
    method, action = routes[goal]
    return {"method": method, "tuner": "lora", "why": action,
            "next": "Prepare independent train/eval JSONL, inspect examples, then plan. LoRA is a weight-update method, not a loss function.",
            "limits": "No algorithm guarantees improved quality. Compare against the unchanged model on held-out tasks."}


def scan_model(path):
    root = Path(path).expanduser().resolve(strict=True)
    cfg = json.loads((root / "config.json").read_text())
    if cfg.get("model_type") not in TEXT_TYPES or cfg.get("auto_map") or cfg.get("vision_config") or cfg.get("quantization_config"):
        raise ValueError("First adapter supports unquantized text Llama/Qwen2/Qwen3/Mistral only; custom code, MoE/multimodal/quantized checkpoints need separate adapters.")
    weights = sorted(root.glob("*.safetensors"))
    if not weights:
        raise ValueError("Local safetensors weights are required; no implicit download or pickle loading")
    index = root / "model.safetensors.index.json"
    if index.exists():
        for name in set(json.loads(index.read_text())["weight_map"].values()):
            if Path(name).name != name or not (root / name).is_file():
                raise ValueError("Missing or unsafe weight shard in index")
    token_files = [root / n for n in ("tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja") if (root / n).exists()]
    if not (root / "tokenizer_config.json").exists() or not any(p.name in {"tokenizer.json", "tokenizer.model", "vocab.json"} for p in token_files):
        raise ValueError("Tokenizer files and tokenizer_config.json are required")
    tc = json.loads((root / "tokenizer_config.json").read_text())
    if not tc.get("chat_template") and not (root / "chat_template.jinja").exists():
        raise ValueError("Explicit chat template required; do not invent one")
    if tc.get("auto_map"):
        raise ValueError("Custom tokenizer code requires a separately reviewed adapter")
    return {"path": str(root), "type": cfg["model_type"], "config": cfg,
            "config_sha256": file_digest(root / "config.json"),
            "weight_index_sha256": file_digest(index) if index.exists() else None,
            "tokenizer_files": {p.name: file_digest(p) for p in token_files},
            "weights_bytes": sum(p.stat().st_size for p in weights),
            "weight_inventory": [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in weights],
            "identity_limit": "Weight inventory is NOT a full weight content digest; use immutable verified snapshots."}


def validate_data(path, method):
    path = Path(path).expanduser().resolve(strict=True)
    rows, hashes, solutions = 0, set(), []
    with path.open("rb") as f:
        while True:
            line = f.readline(8 * 1024 * 1024 + 1)
            if not line:
                break
            if len(line) > 8 * 1024 * 1024:
                raise ValueError("JSONL line exceeds 8 MiB")
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages", [])
            if not messages or any(not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"} or not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
                raise ValueError(f"Invalid text messages at data row {rows + 1}")
            dialog = messages[1:] if messages[0]["role"] == "system" else messages
            if not dialog or any(m["role"] != ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(dialog)):
                raise ValueError("Require optional leading system, then alternating user/assistant messages")
            end = "assistant" if method in {"sft", "dpo", "rm"} else "user"
            if messages[-1]["role"] != end:
                raise ValueError(f"{method} messages must end with {end}")
            if method in {"dpo", "rm"}:
                rejected = row.get("rejected_response")
                if not isinstance(rejected, str) or not rejected.strip() or rejected == messages[-1]["content"]:
                    raise ValueError("Preference pairs need a nonempty rejected_response different from chosen")
            if method == "grpo":
                if not isinstance(row.get("solution"), str) or not row["solution"].strip():
                    raise ValueError("Math GRPO needs a nonempty solution for the accuracy reward")
                solutions.append(row["solution"])
            allowed = {"messages"} | ({"solution"} if method == "grpo" else {"rejected_response"} if method in {"dpo", "rm"} else set())
            if set(row) - allowed:
                raise ValueError(f"Unexpected dataset fields: {sorted(set(row) - allowed)}")
            # Compare prompts, not answers, to catch train/eval leakage across variants.
            prompt = messages[:-1] if end == "assistant" else messages
            hashes.add(digest(prompt))
            rows += 1
    if not rows:
        raise ValueError("Dataset is empty")
    return {"path": str(path), "rows": rows, "sha256": file_digest(path), "prompt_hashes": sorted(hashes), "solutions": solutions}


def build_plan(spec):
    known = {"model", "train", "eval", "method", "tuner", "python", "output", "gpus", "max_length", "steps", "batch", "accumulation", "lr", "rank", "generations", "completion_length", "reward_model"}
    if set(spec) - known:
        raise ValueError(f"Unknown settings: {sorted(set(spec) - known)}")
    method, tuner = spec["method"], spec.get("tuner", "lora")
    if method not in METHODS or tuner not in {"lora", "full"}:
        raise ValueError("Unknown method or tuner")
    defaults = {"gpus": 1, "max_length": 2048, "steps": 20, "batch": 1,
                "accumulation": {"grpo": 4, "ppo": 2}.get(method, 16), "lr": 1e-4 if method in {"sft", "rm"} and tuner == "lora" else 1e-5,
                "rank": 8, "generations": 4, "completion_length": 256, "tuner": tuner}
    s = {**defaults, **spec}
    for key in ("gpus", "max_length", "steps", "batch", "accumulation", "rank", "generations", "completion_length"):
        if type(s[key]) is not int or s[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if s["gpus"] > 8 or s["max_length"] > 32768 or s["steps"] > 1000000 or s["batch"] > 1024 or s["accumulation"] > 65536:
        raise ValueError("Outside first adapter bounds: <=8 GPUs, <=32768 tokens, <=1M steps, batch<=1024, accumulation<=65536")
    if not isinstance(s["lr"], (int, float)) or not math.isfinite(s["lr"]) or not 0 < s["lr"] < 1:
        raise ValueError("lr must be finite and between 0 and 1")
    model = scan_model(s["model"])
    context = model["config"].get("max_position_embeddings")
    if isinstance(context, int) and s["max_length"] > context:
        raise ValueError("Length exceeds checkpoint context; extension needs a separate recipe")
    data, evaluation = validate_data(s["train"], method), validate_data(s["eval"], method)
    if set(data["prompt_hashes"]) & set(evaluation["prompt_hashes"]):
        raise ValueError("Training/evaluation prompts overlap; prepare independent held-out examples")
    output = Path(s["output"]).expanduser().resolve()
    if output.exists():
        raise ValueError("Use a fresh output directory; never overwrite a previous experiment")
    for protected in (Path(model["path"]), Path(data["path"]), Path(evaluation["path"])):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("Output must be separate from model and dataset paths")
    python = Path(s["python"]).expanduser().absolute()
    if not python.is_file():
        raise ValueError("Provide an existing trusted backend Python executable")
    s.update(model=model["path"], train=data["path"], eval=evaluation["path"], output=str(output), python=str(python))
    argv = [str(python)]
    module = "swift.cli.sft" if method == "sft" else "swift.cli.rlhf"
    if s["gpus"] > 1:
        argv += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(s["gpus"]), "--module", module]
    else:
        argv += ["-m", module]
    flags = {"model": s["model"], "dataset": s["train"], "val_dataset": s["eval"], "split_dataset_ratio": 0,
             "tuner_type": tuner, "torch_dtype": "bfloat16", "max_steps": s["steps"], "max_length": s["max_length"],
             "per_device_train_batch_size": s["batch"], "per_device_eval_batch_size": s["batch"],
             "gradient_accumulation_steps": s["accumulation"], "learning_rate": s["lr"],
             "logging_steps": 1, "eval_steps": min(10, s["steps"]), "save_steps": min(10, s["steps"]),
             "save_total_limit": 2, "output_dir": str(output / "artifacts"), "report_to": "none", "seed": 42,
             "gradient_checkpointing": "true", "dataloader_num_workers": 0}
    if tuner == "lora":
        flags.update(lora_rank=s["rank"], lora_alpha=s["rank"] * 2, target_modules="all-linear")
    if s["gpus"] > 1:
        flags["deepspeed"] = "zero2" if tuner == "lora" else "zero3"
    if method != "sft":
        flags["rlhf_type"] = method
    roles = ["trainable policy" if method != "rm" else "trainable scalar reward model"]
    reward = None
    if method in {"dpo", "grpo", "ppo"}:
        roles.append("reference via disabled policy LoRA adapter" if tuner == "lora" else "separate frozen reference policy")
    if method == "dpo":
        flags["beta"] = .1
    if method in {"grpo", "ppo"}:
        if s["completion_length"] >= s["max_length"]:
            raise ValueError("Completion budget must leave room for the prompt")
        flags.update(max_completion_length=s["completion_length"], temperature=.9 if method == "grpo" else .7)
    if method == "grpo":
        total = s["gpus"] * s["batch"] * s["accumulation"]
        if s["generations"] < 2 or total % s["generations"]:
            raise ValueError("GRPO requires >=2 generations and (GPUs × batch × accumulation) divisible by generations")
        flags.update(reward_funcs="accuracy", num_generations=s["generations"], num_generations_eval=1, use_vllm="false", beta=.04)
        if data["rows"] < total // s["generations"] or evaluation["rows"] < s["gpus"] * s["batch"]:
            raise ValueError("GRPO datasets must contain at least one complete train/eval global prompt batch")
        roles += ["in-process Transformers rollout (no separate vLLM GPU)", "math accuracy reward; no critic"]
    if method == "ppo":
        if not s.get("reward_model"):
            raise ValueError("PPO requires a local scalar reward checkpoint; first train RM from preference pairs or supply a compatible one")
        rollout_batch = s["gpus"] * s["batch"] * s["accumulation"]
        if data["rows"] < rollout_batch or evaluation["rows"] < s["gpus"] * s["batch"]:
            raise ValueError("PPO drop-last loaders require at least a full rollout batch of training prompts and a full evaluation batch")
        reward = scan_model(s["reward_model"])
        cfg = reward["config"]
        if not any("SequenceClassification" in a for a in cfg.get("architectures", [])) or not (cfg.get("num_labels") == 1 or len(cfg.get("id2label", {})) == 1):
            raise ValueError("Reward must be a scalar SequenceClassification checkpoint, not a chat model or unmerged adapter")
        if reward["tokenizer_files"] != model["tokenizer_files"]:
            raise ValueError("PPO first adapter requires identical tokenizer/template files for policy and reward")
        flags.update(reward_model=reward["path"], local_rollout_forward_batch_size=1)
        # TRL 0.26 PPO sizes its loop from int(epochs * dataset_rows), not max_steps.
        # A half-episode margin avoids floating-point rounding to one episode less.
        flags.pop("max_steps")
        flags["num_train_epochs"] = (s["steps"] * rollout_batch + .5) / data["rows"]
        s["reward_model"] = reward["path"]
        roles += ["frozen scalar reward model", "separate trainable value model initialized from reward checkpoint", "in-process Transformers rollout"]
    for k, value in flags.items():
        argv += ["--" + k, str(value)]
    result = {"schema": 1, "backend": f"ms-swift=={BACKEND_VERSION}; trl=={TRL_VERSION}", "spec": s,
              "model": model, "reward_model": reward, "train_data": data, "eval_data": evaluation,
              "argv": argv, "roles": roles, "parameters": flags,
              "decision": [advise({"sft": "examples", "dpo": "preferences", "rm": "train-reward", "grpo": "math", "ppo": "reward-model"}[method]),
                           {"weight_updates": tuner, "precision": "BF16 weights/compute; optimizer state is backend-dependent FP32, not BF16 everywhere"},
                           {"parallel": "single GPU" if s["gpus"] == 1 else f"single-node DP {s['gpus']} with {flags['deepspeed']}; conservative recipe, not an optimal topology search"}],
              "validation": "Command contract only; no real GPU training has been validated by this release",
              "warnings": ["Weight file bytes are not peak VRAM. Activations, optimizer, reference/value/reward roles and rollout KV must also fit.",
                           "20 steps is a pipeline smoke test, not a quality-training recommendation; PPO counts rollout/update cycles, each with internal PPO epochs.",
                           "Held-out loss/reward is not a complete quality evaluation. Compare behavior and reward hacking against the base model.",
                           "Local checkpoints/datasets are required. Backend network is set offline; audit logs may contain private data."]}
    result["approval_sha256"] = digest(result)
    return result
