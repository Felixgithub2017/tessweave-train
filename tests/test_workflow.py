import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tessweave_train.core import advise, build_plan, validate_data
from tessweave_train.runtime import evaluate_probe, launch_env, revalidate, run


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.config = {"model_type": "qwen2", "max_position_embeddings": 4096}
        self.write_config(self.config)
        # Metadata-only fixtures, NOT runnable fake weights or a training test.
        (self.model / "model.safetensors").write_bytes(b"test fixture")
        (self.model / "tokenizer.json").write_text("{}")
        (self.model / "tokenizer_config.json").write_text(json.dumps({"chat_template": "fixture"}))

    def write_config(self, config):
        (self.model / "config.json").write_text(json.dumps(config))

    def spec(self, method="sft"):
        fixtures = Path(__file__).parents[1] / "examples"
        kind = "dpo" if method == "rm" else method
        return dict(method=method, model=str(self.model), python=sys.executable,
                    train=str(fixtures / f"{kind}.train.jsonl"), eval=str(fixtures / f"{kind}.eval.jsonl"), output=str(self.root / "run"))

    def test_goal_routes_and_lora_is_orthogonal(self):
        for goal, method in [("examples", "sft"), ("preferences", "dpo"), ("math", "grpo"), ("reward-model", "ppo")]:
            self.assertEqual(advise(goal)["method"], method)
            self.assertEqual(advise(goal)["tuner"], "lora")

    def test_guided_wizard_builds_reviewable_plan(self):
        import contextlib
        import io
        from tessweave_train.cli import wizard, save_plan
        s = self.spec()
        answers = ["1", s["model"], s["train"], s["eval"], s["python"], "1", "1024", s["output"]]
        with patch("builtins.input", side_effect=answers), contextlib.redirect_stdout(io.StringIO()):
            p = wizard()
            dest = self.root / "plan.json"
            save_plan(p, dest)
            self.assertEqual(json.loads(dest.read_text())["approval_sha256"], p["approval_sha256"])
            with self.assertRaises(FileExistsError): save_plan(p, dest)
        self.assertEqual(p["spec"]["method"], "sft")
        self.assertFalse(Path(s["output"]).exists())

    def test_real_backend_argv_all_primary_recipes(self):
        for method in ("sft", "dpo", "grpo", "rm"):
            p = build_plan(self.spec(method))
            self.assertIn("swift.cli.sft" if method == "sft" else "swift.cli.rlhf", p["argv"])
            self.assertEqual(p["parameters"]["tuner_type"], "lora")
            self.assertEqual(p["parameters"]["split_dataset_ratio"], 0)
            self.assertEqual(revalidate(p, p["approval_sha256"]), p)

    def test_full_multi_gpu_roles(self):
        p = build_plan({**self.spec("dpo"), "gpus": 2, "tuner": "full"})
        self.assertIn("torch.distributed.run", p["argv"])
        self.assertEqual(p["parameters"]["deepspeed"], "zero3")
        self.assertIn("separate frozen reference policy", p["roles"])

    def test_ppo_requires_reward(self):
        with self.assertRaisesRegex(ValueError, "scalar reward"):
            build_plan(self.spec("ppo"))

    def test_ppo_role_and_reward_contract(self):
        import shutil
        reward = self.root / "reward"
        shutil.copytree(self.model, reward)
        (reward / "config.json").write_text(json.dumps({**self.config, "architectures": ["Qwen2ForSequenceClassification"], "num_labels": 1}))
        spec = {**self.spec("ppo"), "reward_model": str(reward)}
        p = build_plan(spec)
        self.assertIn("separate trainable value model initialized from reward checkpoint", p["roles"])
        self.assertEqual(p["parameters"]["local_rollout_forward_batch_size"], 1)
        self.assertNotIn("max_steps", p["parameters"])
        episodes = int(p["parameters"]["num_train_epochs"] * p["train_data"]["rows"])
        self.assertEqual(episodes, p["spec"]["steps"] * p["spec"]["accumulation"])
        (reward / "tokenizer.json").write_text('{"different": true}')
        with self.assertRaisesRegex(ValueError, "identical tokenizer"):
            build_plan(spec)

    def test_architecture_custom_code_quantized_rejected(self):
        for delta in ({"model_type": "qwen3_moe"}, {"auto_map": {"AutoModel": "evil.Model"}}, {"quantization_config": {"bits": 4}}):
            self.write_config({**self.config, **delta})
            with self.assertRaises(ValueError): build_plan(self.spec())

    def test_missing_shard(self):
        (self.model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "missing.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "shard"): build_plan(self.spec())

    def test_no_silent_data_leak(self):
        s = self.spec()
        with self.assertRaisesRegex(ValueError, "overlap"): build_plan({**s, "eval": s["train"]})

    def test_invalid_preferences_and_turn_order(self):
        path = self.root / "bad.jsonl"
        for row in ({"messages": [{"role": "assistant", "content": "x"}]},
                    {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], "rejected_response": "a"}):
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError): validate_data(path, "dpo")

    def test_grpo_group_and_length_constraints(self):
        for delta in ({"generations": 1}, {"generations": 3}, {"max_length": 128}):
            with self.assertRaises(ValueError): build_plan({**self.spec("grpo"), **delta})
        p = build_plan(self.spec("grpo"))
        self.assertEqual(p["parameters"]["num_generations_eval"], 1)
        with self.assertRaisesRegex(ValueError, "global prompt batch"):
            build_plan({**self.spec("grpo"), "gpus": 2})

    def test_bounds_finite_and_fresh_output(self):
        for delta in ({"lr": float("nan")}, {"gpus": 0}, {"steps": True}, {"max_length": 8192}, {"shell": "bad"}, {"output": str(self.model)}):
            with self.assertRaises(ValueError): build_plan({**self.spec(), **delta})

    def test_approval_and_changed_inputs(self):
        p = build_plan(self.spec())
        with self.assertRaises(ValueError): revalidate(p, "yes")
        tampered = copy.deepcopy(p)
        tampered["argv"] += ["--bad"]
        with self.assertRaises(ValueError): revalidate(tampered, p["approval_sha256"])
        self.write_config({**self.config, "changed": 1})
        with self.assertRaisesRegex(ValueError, "changed"): revalidate(p, p["approval_sha256"])

    def test_environment_is_rejected_actionably(self):
        p = build_plan(self.spec())
        r = evaluate_probe(p, {"platform": "Darwin", "packages": {}, "gpus": [], "errors": []})
        self.assertFalse(r["ready_for_smoke_run"])
        self.assertTrue(any("Mac" in b for b in r["blockers"]))

    def test_selected_gpu_mask(self):
        p = build_plan(self.spec())
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-abc,GPU-def"}):
            self.assertEqual(launch_env(p)["CUDA_VISIBLE_DEVICES"], "GPU-abc")

    def test_blocked_run_creates_no_training_directory(self):
        p = build_plan(self.spec())
        with patch("tessweave_train.runtime.doctor", return_value={"blockers": ["No GPU"]}):
            with self.assertRaisesRegex(ValueError, "No GPU"): run(p, p["approval_sha256"])
        self.assertFalse((self.root / "run").exists())

    def test_runner_logs_subprocess_and_failure_honestly(self):
        # Substitute only the subprocess boundary. No training performance claim.
        p = build_plan(self.spec())
        from unittest.mock import MagicMock
        proc = MagicMock(stdout=iter(["loss=1.2\n"]), pid=123)
        proc.wait.return_value = 7
        with patch("tessweave_train.runtime.doctor", return_value={"blockers": []}), \
             patch("tessweave_train.runtime.subprocess.Popen", return_value=proc), \
             patch("tessweave_train.runtime.subprocess.run", side_effect=FileNotFoundError("no nvidia-smi")):
            self.assertEqual(run(p, p["approval_sha256"]), 7)
        root = self.root / "run"
        self.assertEqual(json.loads((root / "status.json").read_text())["state"], "failed")
        self.assertIn("loss=1.2", (root / "console.log").read_text())
        self.assertIn('"kind": "finish"', (root / "events.jsonl").read_text())


if __name__ == "__main__": unittest.main()
