"""Explicit approval, revalidation, backend probing and terminal audit logs."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import threading
import time

from .core import BACKEND_VERSION, TRL_VERSION, build_plan, digest

OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "MODELSCOPE_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}


def launch_env(plan):
    env = {**os.environ, **OFFLINE}
    n = plan["spec"]["gpus"]
    visible = env.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else [str(i) for i in range(n)]
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices[:n])
    return env

# Executed only in the explicitly selected trusted environment. No weights load.
PROBE = r'''
import importlib, importlib.metadata as md, json, platform, sys
p = json.load(sys.stdin)
out = {"platform": platform.system(), "python": sys.version, "packages": {}, "errors": [], "gpus": []}
for name in ("ms-swift", "torch", "transformers", "peft", "trl", "deepspeed", "math-verify"):
    try: out["packages"][name] = md.version(name)
    except md.PackageNotFoundError: out["packages"][name] = None
try:
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            with torch.cuda.device(i): bf16 = torch.cuda.is_bf16_supported()
            out["gpus"].append({"name": props.name, "total_bytes": props.total_memory, "bf16": bf16})
except Exception as e: out["errors"].append("GPU probe: " + str(e))
try:
    method = p["spec"]["method"]
    importlib.import_module("swift.pipelines.train.sft" if method == "sft" else "swift.pipelines.train.rlhf")
    if method != "sft":
        getattr(importlib.import_module("swift.rlhf_trainers"), {"dpo": "DPOTrainer", "rm": "RewardTrainer", "grpo": "GRPOTrainer", "ppo": "PPOTrainer"}[method])
    out["backend_import"] = True
except Exception as e: out["errors"].append("Backend import: " + str(e))
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(p["spec"]["model"], local_files_only=True, trust_remote_code=False)
    max_seen, rows = 0, 0
    for split in ("train_data", "eval_data"):
        with open(p[split]["path"]) as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                dialogs = [row["messages"]]
                if "rejected_response" in row:
                    dialogs.append(row["messages"][:-1] + [{"role": "assistant", "content": row["rejected_response"]}])
                for messages in dialogs:
                    generating = p["spec"]["method"] in {"ppo", "grpo"}
                    n = len(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=generating))
                    n += p["spec"]["completion_length"] if generating else 0
                    max_seen = max(max_seen, n)
                    if n > p["spec"]["max_length"]: raise ValueError("Example plus completion budget exceeds max_length; shorten/split data or explicitly increase length")
                if p["spec"]["method"] == "grpo":
                    from math_verify import parse
                    if not parse(row["solution"], extraction_mode="first_match"):
                        raise ValueError("Unparseable math solution would yield a zero reward")
                rows += 1
    out["token_check"] = {"rows": rows, "max_tokens_with_completion_budget": max_seen, "template": "checkpoint AutoTokenizer; verify ms-swift masking in first smoke run"}
except Exception as e: out["errors"].append("Tokenizer/data check: " + str(e))
print("TESSWEAVE_PROBE=" + json.dumps(out))
'''


def evaluate_probe(plan, report):
    blockers = list(report.get("errors", []))
    if report.get("platform") != "Linux":
        blockers.append("First training adapter requires Linux + NVIDIA CUDA. Mac can prepare/validate plans; do not install CUDA on macOS.")
    if report.get("packages", {}).get("ms-swift") != BACKEND_VERSION:
        blockers.append(f"Use the reviewed ms-swift=={BACKEND_VERSION} environment, not an arbitrary current release")
    if report.get("packages", {}).get("trl") != TRL_VERSION:
        blockers.append(f"Use trl=={TRL_VERSION}; GRPO/PPO batch and duration contracts are version-specific")
    gpus = report.get("gpus", [])
    if len(gpus) < plan["spec"]["gpus"]:
        blockers.append("Insufficient visible CUDA GPUs; inspect CUDA_VISIBLE_DEVICES and driver/container GPU access")
    if any(not gpu["bf16"] for gpu in gpus[:plan["spec"]["gpus"]]):
        blockers.append("This recipe requires BF16-capable GPUs; GTX 1070 is not supported by this adapter")
    if plan["spec"]["gpus"] > 1 and not report.get("packages", {}).get("deepspeed"):
        blockers.append("Multi-GPU recipe requires DeepSpeed")
    if "token_check" not in report:
        blockers.append("Tokenizer/data token-length check did not complete")
    if not report.get("backend_import"):
        blockers.append("Selected trainer could not be imported in the backend environment")
    return {**report, "blockers": blockers, "ready_for_smoke_run": not blockers,
            "scope": "No weights loaded and no peak-VRAM/quality guarantee. First GPU run must verify actual memory and behavior."}


def doctor(plan):
    proc = subprocess.run([plan["spec"]["python"], "-I", "-c", PROBE],
                          input=json.dumps(plan), text=True, capture_output=True,
                          timeout=300, env=launch_env(plan))
    lines = [line for line in proc.stdout.splitlines() if line.startswith("TESSWEAVE_PROBE=")]
    if proc.returncode or not lines:
        raise ValueError("Environment probe failed: " + (proc.stderr or proc.stdout)[-4000:])
    return evaluate_probe(plan, json.loads(lines[-1].split("=", 1)[1]))


def revalidate(plan, approval):
    unsigned = {k: v for k, v in plan.items() if k != "approval_sha256"}
    if approval != plan.get("approval_sha256") or digest(unsigned) != approval:
        raise ValueError("Approval must match the complete, unmodified plan SHA256")
    fresh = build_plan(plan["spec"])
    if fresh["approval_sha256"] != approval:
        raise ValueError("Inputs, recipe or files changed since review; rebuild and review the plan")
    return fresh


def stop_process(process, first_signal=signal.SIGINT):
    if process is None or process.poll() is not None:
        return
    for sig, timeout in ((first_signal, 20), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue


def run(plan, approval):
    plan = revalidate(plan, approval)
    preflight = doctor(plan)
    if preflight["blockers"]:
        raise ValueError("Preflight blocked:\n" + "\n".join(preflight["blockers"]))
    root = Path(plan["spec"]["output"])
    root.mkdir(parents=True, exist_ok=False)
    (root / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    (root / "preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2))
    (root / "command.txt").write_text(shlex.join(plan["argv"]) + "\n")
    started = time.time()
    code, interrupted, process = None, False, None
    status = {"state": "starting", "started_at_unix": started}
    def save_status():
        (root / "status.json").write_text(json.dumps(status, indent=2))
    save_status()
    stop_sampling = threading.Event()
    def sample_gpu():
        command = ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw", "--format=csv,noheader,nounits"]
        with (root / "gpu.jsonl").open("w") as f:
            while not stop_sampling.is_set():
                row = {"time_unix": time.time(), "argv": command, "scope": "whole GPU, not exclusively this job; memory MiB, power W"}
                try:
                    sample = subprocess.run(command, text=True, capture_output=True, timeout=3)
                    row.update(returncode=sample.returncode, stdout=sample.stdout, stderr=sample.stderr)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    row["unavailable"] = str(exc)
                f.write(json.dumps(row) + "\n")
                f.flush()
                stop_sampling.wait(2)
    sampler = threading.Thread(target=sample_gpu, daemon=True)
    sampler.start()
    with (root / "console.log").open("w") as log, (root / "events.jsonl").open("w") as events:
        def event(kind, **data):
            events.write(json.dumps({"time_unix": time.time(), "kind": kind, **data}, ensure_ascii=False) + "\n")
            events.flush()
        env = launch_env(plan)
        event("launch", argv=plan["argv"], environment_overrides={**OFFLINE, "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"]})
        try:
            # No shell interpolation; the user watches the real trainer in this terminal.
            process = subprocess.Popen(plan["argv"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1,
                                       env=env, start_new_session=True)
            status.update(state="running", pid=process.pid)
            save_status()
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                event("backend_output", text=line)
            code = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            if process is not None:
                stop_process(process)
                code = process.returncode
        except Exception as exc:
            stop_process(process, signal.SIGTERM)
            event("runner_error", message=str(exc))
            raise
        finally:
            stop_sampling.set()
            sampler.join(timeout=5)
            status.update(state="interrupted" if interrupted else "completed" if code == 0 else "failed",
                          exit_code=code, elapsed_s=time.time() - started)
            save_status()
            event("finish", **status)
    return code if code is not None else 130
