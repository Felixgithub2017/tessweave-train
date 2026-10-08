# 从训练目标走到第一条真实训练命令

TessWeave Train 的目标是让系统引导你完成训练，而不是让你先学会所有框架参数。当前是第一条 CLI 实现链：指导选算法、校验本地数据、生成可审阅命令、检查环境、经确认后启动真实 ms-swift 训练、保留全过程记录。尚未通过 GPU 实训认证，也还没有接入 Studio 图形界面。

## 第一步 先判断你给模型什么反馈

运行 `tessweave-train guide`。不要从“我想用最新的 RL 算法”开始，而要从手中监督信号开始。

如果你有一批正确回答，先做 **SFT**：模型看见问题，在正确回答的 token 上计算训练损失。LoRA 让大部分原权重冻结，只更新低秩增量；它不是另一种任务，所以存在 SFT+LoRA、DPO+LoRA、GRPO+LoRA。

如果你能比较两个回答哪个更好，考虑 **DPO**：同一个问题，提供 chosen/rejected 两个答案。训练目标相对于参考策略提高被选答案的偏好。它不需要训练过程在线生成海量答案，也不需要额外标量奖励模型。

如果你能自动验算结果，比如数学题，考虑 **GRPO**：同一个问题采样多个答案，用奖励区分它们，再用组内相对信号更新策略。没有可辨别的奖励，采样再多也没有可靠学习方向。第一版只支持数学 `accuracy` 奖励，不把它冒充通用任务奖励。

如果你有可靠的标量奖励模型，才进入 **PPO**：策略生成回答，奖励模型打分，价值模型估计回报，策略与价值模型一起更新。参考策略约束漂移。角色多，显存与调试成本也多。仅有聊天模型不能当作标量奖励模型。

如果你只有偏好数据、没有奖励模型，可以先做 **RM**。RM 的 LoRA 产物还需要经检查导出为完整标量奖励 checkpoint，再用于本项目 PPO；第一版不自动串联导出、奖励评估和 PPO。

## 第二步 在真正训练的服务器准备环境

你当前的 M5 Mac 用来写数据、检查方案、阅读日志。第一条训练适配器在 Linux NVIDIA 上执行；不要在 Mac 的环境里安装 vLLM/CUDA 来强行满足它。

以下是**需要你确认再手动执行**的候选环境命令。优先使用云服务商已提供、与驱动兼容的 PyTorch CUDA 环境；若新建 conda 环境，PyTorch CUDA wheel 必须先按该服务器驱动安装并通过 CUDA 检查。这里不提供一个假装适用于所有驱动的 CUDA 安装命令。

```bash
conda create -n tessweave-train python=3.11 -y
conda activate tessweave-train
# 先安装与该主机驱动匹配的 PyTorch CUDA 构建，或改用已验证的服务商环境。
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
python -m pip install 'ms-swift==4.5.3' 'trl==0.26.0' math-verify
# 仅多 GPU 配方需要；版本范围来自对应上游安装脚本，仍需硬件验证。
# python -m pip install 'deepspeed<0.19'
python -m pip check
python -m pip freeze
python -m pip install -e .
```

`ms-swift` 和 `trl` 固定了适配器目标版本，但其传递依赖尚不是全平台锁文件。第一次成功实训后，应保存完整 freeze、驱动、CUDA、模型 revision 和镜像摘要，作为该硬件组合的验证环境。安装过程可能联网；训练命令不会替你下载权重。下载与代码代理策略仍由你控制。

## 第三步 先用小模型做管线验证

建议第一轮用已下载的 Qwen2.5-0.5B-Instruct 完整 safetensors 目录，避免把环境错误、模型太大和奖励问题混在一起。目录需要 config、权重、tokenizer 和明确的 chat template。这里不宣称已经在你的 H100 上跑通。

编辑 `examples/sft.spec.json`，至少改四类路径：

如果你不想编辑配置文件，可以直接运行 `tessweave-train wizard --out plan.json`。向导会问你掌握哪种监督数据、模型和数据放在哪里、使用哪个 Python 环境、几张 GPU，并解释它建议的算法，最后生成同一种可审阅计划。下面的 JSON 路线适合重复实验和精确修改参数；两条路线任选一条。

- `model`：GPU 服务器上的模型目录，不是 Mac 上的 `/Volumes/...`。
- `python`：真正装了 ms-swift 的环境解释器，不能填工作台系统 Python。
- `train` / `eval`：独立的训练、验证 JSONL 文件。
- `output`：一个尚不存在的新实验目录。

自带两条训练样例和一条验证样例只用来确认格式与执行链，不会教出一个有用模型。准备真实数据时，先抽样人工核对：回答是否正确？系统提示是否一致？验证问题是否在训练集出现过？敏感数据能否出现在日志中？

```bash
tessweave-train plan --spec examples/sft.spec.json --out plan.json
```

预期得到一个计划文件、实际命令、方案 SHA256。此时没有开始训练，也没有下载模型。打开计划看 `decision`、`roles`、`parameters`、`warnings`。SHA256 是该次方案的确认标识，不是登录令牌，也不能证明方案经过性能验证。

```bash
tessweave-train doctor --plan plan.json
```

预期看见实际 Python/依赖版本、GPU 名称与显存总量、BF16 支持、tokenization 检查。`ready_for_smoke_run: true` 表示这些前置检查通过，**不等于峰值显存一定够，也不等于模型质量达标**。失败时先修复明确列出的 blocker；不跳过预检。

```bash
tessweave-train run --plan plan.json --approve 这里替换成完整的方案SHA256
```

当前终端开始运行真实训练，loss 等信息来自后端原始输出。默认 20 步是工程冒烟测试。检查 loss 是否有限、梯度是否异常、checkpoint 是否保存、实际显存峰值是否有余量。Ctrl-C 可以停止。退出码非零会记为失败，不会用“已启动”冒充“训练成功”。

## 第四步 在相同工作流里换训练目标

复制 SFT spec 到新的文件，使用新的 `output` 和计划文件名，再按下表更改；所有方法仍经过同样的 plan、doctor、approve、run。不要复用已经运行过的 output。

| 目标 | 必须修改的配置与数据 |
|---|---|
| DPO | `method: dpo`，数据参考 `examples/dpo.*.jsonl`；用偏好对，不要把 rejected 放进提示词 |
| 数学 GRPO | `method: grpo`，`generations: 4`，`accumulation: 4`，`completion_length: 256`，`lr: 0.00001`；数据参考 `examples/grpo.*.jsonl` |
| PPO | `method: ppo`，`reward_model` 指向完整的标量奖励模型，`accumulation: 2`，`completion_length: 256`，`lr: 0.00001`；提示数据参考 `examples/ppo.*.jsonl` |
| 训练奖励模型 | `method: rm`，使用 DPO 格式偏好对；另做奖励校准和完整 checkpoint 导出后再尝试 PPO |
| 全参数更新 | `tuner: full`，显存角色重新核对，起始学习率改为较保守的值；不要以 LoRA 的显存需求估算全参 |

GRPO 的 `solution` 只供奖励计算，不能塞进用户输入泄露答案。第一版使用上游数学解析器；答案必须可解析，生成回答建议按样例输出 `\\boxed{...}`。奖励全为 0 或全相同，是需要诊断的问题，不是继续延长训练的理由。

GRPO 训练默认每题生成 4 个回答，验证每题生成 1 个回答，避免小验证批次不满足上游整除约束；这不是 pass@4 验证。PPO 的 `steps` 则映射成 rollout/update 循环数，而不是把 SFT 的 `max_steps` 原样传进去；每轮内部还有多次 PPO 更新。数据量不足以组成完整批次时，规划阶段会阻止启动，避免 drop-last 数据加载器空转。

PPO 第一版严格要求 policy 和 reward 的 tokenizer/template 文件一致，reward 必须是单输出 SequenceClassification 模型。这个检查偏保守，但避免不同 token ID 语义导致“看似训练、实际奖励错误”。上游还会从 reward checkpoint 初始化独立的可训练 value model；不能只算一份 policy 的显存。

## 第五步 看懂结果再扩大

`status.json` 回答是否成功退出；`console.log` 回答后端实际说了什么；`gpu.jsonl` 记录整卡利用率、显存和功耗采样，不能当作当前进程独占用量；`events.jsonl` 是带时间戳的输出序列，后续 Studio 可据此做回放。

`artifacts/` 中通常还有 ms-swift 自动创建的版本目录和 checkpoint。LoRA 产物是 adapter，不是完整基础模型。推理时必须使用匹配的基础 checkpoint 加 adapter，或经受控导出合并；本版本没有自动部署这一产物。

扩大路线：单卡小模型跑通 SFT → DPO → GRPO 奖励调试 → 兼容 reward 的 PPO → 更长上下文、更大模型、多卡。每次只改一个主要变量，保存 baseline 与新配置的结果。不是所有任务都需要经过所有算法；系统应帮助你选择最短的有效路径。

最终验收必须包含：未训练模型对照、独立任务集、质量指标、失败案例、速度与成本、奖励投机检查。只看到训练 loss 下降不能宣布“训练得更好”。
