# Continuation-Test Reinforcement Learning

《Learning Reusable Latent Reasoning via Continuation-Test Reinforcement Learning》的可运行研究实现。模型先读前缀，生成一次连续潜状态，再用同一个潜状态回答多个后缀。程序执行器给出训练标签与离线评估答案。

此目录包含训练源码、配置、60,000 个 CT-Bench 训练前缀、公开任务 smoke 样本及实验报告。复跑下方 smoke 命令可生成检查点和完整逐样本记录。运行记录见 [完整 smoke 报告](reports/smoke/REPORT.md)，实验设计与参数选择见 [实验协议与超参数](EXPERIMENTS.md)。开发验证使用 Python 3.12、PyTorch 2.8.0 和 Transformers 4.57.6。

从项目根目录可直接重跑全部实验分支。下面的输出目录适用于第一次运行。重复执行相同命令会从已提交的阶段继续。

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,public]'
python -m ct_rl.paper --profile smoke --out runs/my-smoke
python -m pytest -q
~~~

smoke 使用包内的小模型配置与真实公开样本，安装依赖后可离线运行。正式实验命令为 `python -m ct_rl.paper --profile paper --out runs/paper`。首次执行正式配置时需要访问指定版本的骨干和公开数据。

## 已运行的 smoke

CPU 上的完整运行耗时 9 分 29 秒。19 种方法完成三个种子的等步数与等算力比较，CT-RL 和验证集入选的三个基线完成第二规模检查。公开部分完成 69 次独立适配。51 项测试通过。审计重算了 2,746 条受控配对得分和 483 条公开答案，并逐条核验参考答案与结束符可在预算内输出。同目录复跑后 252 份检查点的摘要与修改时间一致。[运行回执](reports/smoke/execution.json) · [独立复核](reports/smoke/audit.json)

等算力矩阵中，CT-RL 的 ID CT-Acc 三种子均值为 0.00%，Pair-Both 为 0.00%。四个公开 smoke 测试集各取 3 题，主方法的 Pass@1 均为 0。三个训练任务的公开验证集均有多个方法并列第一，入选的零合成对照按方法名确定。另一个 300 步冷启动的主方法示例在 18 个 ID 前缀上得到 18.06% 的 CT-Acc。正式 1.5B／3B 训练效果待测。[分项报告](reports/smoke/REPORT.md)

## 从零跑通

在本项目目录中运行。训练入口支持 Linux 和 macOS，Python 需要 3.10 或以上。下面使用 Python 3.12。

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m ct_rl demo --out runs/demo
python -m pytest -q
~~~

demo 用随机初始化的两层 Qwen2 和内置字节分词器在 CPU 上运行。安装依赖后可以离线执行。它生成三类数据，做 300 步答案监督冷启动、3 轮交替训练，再评估验证集及 5 类测试划分。可直接打开 `runs/demo/evaluation.json` 查看指标、逐样本预测与成本。`runs/demo/train/checkpoint.pt` 可以用于推理或续训。

~~~bash
python -m ct_rl infer \
  --checkpoint runs/demo/train/checkpoint.pt \
  --prefix 'arithmetic a=2 b=3; swap' \
  --suffix left --suffix diff
~~~

这条命令用同一个 `Z` 回答两个后缀，正确答案分别为 3 和 1。`infer` 会读取检查点的推理模式。Direct 和 CoT 使用完整题面，CoT 输出提取后的最终答案。公开任务的原生潜方法使用适配时保存的读出模式。输出 JSON 中的 `mode` 可用于核对。推理默认沿用检查点内的回答上限，读取原先只保存 12 token 的 CoT 检查点时可加 `--tokens 128`。

## 代码与论文的对应关系

| 论文部分 | 可执行实现 |
|---|---|
| 第 3.1 对角高斯策略、连续接口和独立续算上下文 | [model.py](ct_rl/model.py) 中的 `sample`、`replay`、`reader_inputs` 和 `decode` |
| 第 3.2 多轨迹共享后缀、任务奖励 | [curriculum.py](ct_rl/curriculum.py) 中的 `make_tests` 和 `utilities` |
| 第 3.2 留一基线、KL、策略更新和答案序列监督 | [learning.py](ct_rl/learning.py) 中的 `group_update` 和 `supervised_step` |
| 第 3.3 区分性后缀搜索、失败率筛选和难例对照 | [tasks.py](ct_rl/tasks.py) 和 [curriculum.py](ct_rl/curriculum.py) |
| 冷启动、交替训练、下一轮使用反例、断点恢复 | [experiment.py](ct_rl/experiment.py) |
| CT-Bench 划分、执行标签和数据指纹 | [tasks.py](ct_rl/tasks.py) 和 [settings.py](ct_rl/settings.py) |
| CT-Acc、All-Correct、Pair-Both、潜状态干预和开销 | [evaluation.py](ct_rl/evaluation.py) |
| 多随机种子、等步数和算力预算对照 | [suite.py](ct_rl/suite.py) |
| 四个公开基准的读取、适配和 Pass@1 | [public.py](ct_rl/public.py) |
| 五个文献基线、文本参照与循环骨干 | [comparisons.py](ct_rl/comparisons.py) 与 [comparison_learning.py](ct_rl/comparison_learning.py) |
| 原生公开协议、验证选模、两阶段算力匹配 | [paper.py](ct_rl/paper.py)、[paper_transfer.py](ct_rl/paper_transfer.py) 与 [public_data.py](ct_rl/public_data.py) |
| 批量 1 的推理算力档位 | [inference_budget.py](ct_rl/inference_budget.py) |
| 已有第三方权重的统一推理与评分 | [external.py](ct_rl/external.py) 及 `export-tests`、`score-external` 命令 |

## 训练协议与实现选择

默认策略更新对应草稿中的 RLOO 写法。一次策略更新为每个前缀采样 `K` 条轨迹，然后抽取 `M` 个后缀。同一前缀的所有轨迹共享这组后缀和一个冻结的续算器。每条轨迹的任务回报是 `M` 次贪心续算的执行正确率。[RLOO](https://aclanthology.org/2024.acl-long.662/)

若 `old` 是本批采样策略、`ref` 是冷启动后的冻结快照，第 $i$ 条轨迹的回报和优势为

$$
\widetilde R_i = \widehat R_i^{\rm task} -
\beta\big[\log\pi_{\rm old}(U_i|p)-\log\pi_{\rm ref}(U_i|p)\big],
\qquad
A_i=\widetilde R_i-\frac{1}{K-1}\sum_{j\ne i}\widetilde R_j .
$$

代码将已采样的动作固定，重新计算每一步所有动作坐标的联合对数密度，再对步数求和。RLOO 对这批数据更新一次。选择 `algorithm=grpo` 后，代码按组内任务回报计算标准化优势，用每步的联合密度比进行裁剪，并在采样历史上单独计算当前高斯分布相对参考分布的解析 KL。`grpo_epochs` 控制这批轨迹的更新次数。[DeepSeekMath 第 4.1 节](https://arxiv.org/html/2402.03300v3)

草稿留待确定的部分，在本实现中采用以下设定。

- 生成器和续算器共用冻结的 Qwen 骨干，各自使用独立的 Q/V LoRA 参数。生成器另有高斯参数头，续算器有独立的输出参数。小模型训练完整输出头，预训练模型训练原词表输出头上的低秩增量。CT-RL 主方法关闭 dropout，SLPO 在采样潜状态和估计代理分布时开启 MC dropout。[Qwen2 模型接口](https://huggingface.co/docs/transformers/v4.57.1/model_doc/qwen2)
- 模型把低维动作 $u_t$ 经线性映射变为 $z_t$，将它输入下一潜步骤。监督冷启动通过可重参数化采样联合训练两侧及映射。随后 RL 冻结映射和参考策略。
- 策略更新完成后，为答案监督重新生成本轮策略的潜状态。续算器用停止梯度的 `Z`、后缀和正确答案的历史 token 做 teacher forcing。NLL 在答案和 EOS 上求和，再对答案样本取均值。
- 普通续算从固定任务说明、`Z` 和后缀新建上下文。缓存只在同一次生成或续算内使用。每个测试使用一次贪心解码，达到 token 上限仍未输出 EOS 或输出格式错误都计失败。
- 固定种子决定每轮待搜索的训练前缀对。自第二轮起，前缀抽样在全部训练对与上轮预选的前缀对之间各取 50% 的概率。相同种子的控制组共享这个前缀抽样策略。对有反例的前缀，以 `rho` 的概率从池中采一个后缀，其余按基础分布采样。
- 反例搜索在每轮两侧参数更新后执行。两个前缀接受同一候选后缀，用执行器验证两侧标签不同，再用模型在两侧的平均失败率排序。在相同操作数内保留难例，记录两侧标签、失败率及评分轮数。反例从下一轮开始使用。反例池的键对应训练文件里前缀对的 `group`，可由 `Pair.group` 重建。

## CT-Bench 与实验配置

训练与验证的参数单位是**每个任务族的前缀对数**。一对包含两个不同前缀。以下命令按草稿的计划生成每族 20,000 个训练前缀和 1,000 个验证前缀。

~~~bash
python -m ct_rl data --out data/paper \
  --train-pairs 10000 --valid-pairs 500 --eval-pairs 500 --seed 120

python -m ct_rl train --data data/paper \
  --config configs/qwen1.5b.json --out runs/qwen15b --count-flops

python -m ct_rl eval --data data/paper \
  --checkpoint runs/qwen15b/checkpoint.pt --out runs/qwen15b/heldout.json \
  --split id comp prefix_length suffix_length scale --m 4 --bootstrap 1000
~~~

两个正式配置均使用 CUDA、BF16，已锁定模型提交并显式列出训练超参数。首次加载需要可访问的模型权重，也可指定 Hugging Face 格式的本地模型目录。运行记录会保存所加载的版本。[1.5B 配置](configs/qwen1.5b.json) · [3B 配置](configs/qwen3b.json) · [1.5B 模型](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) · [3B 模型](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

| 任务族 | 前缀及基础读出 | 后缀举例 | 单独考察的外推 |
|---|---|---|---|
| 算术 | 两个整数寄存器上的程序，基础读出为和 | `swap ; left` 先交换再读左寄存器 | 前缀从 4–8 步到 12–20 步，后缀从 1–4 个操作到 5–8 个 |
| 图遍历 | 两种有向边上的定长行走，基础读出为终点编号的奇偶性 | `R ; node` 沿 R 边再读编号 | 分开改变行走长度、后缀长度、图节点数 |
| 规则推理 | 一元 Horn 蕴含与初始事实的闭包，基础读出为事实数 | `add 4 ; has 7` 加入事实 4、求闭包再查询事实 7 | 分开改变有效链深度、后缀长度、规则数量 |

这里将草稿中的任务族具体化为上述三种可执行语言。算术后缀的 `raise` 表示左寄存器加 2，`mix` 表示左寄存器加上右寄存器，`neg` 表示右寄存器取负。`sum`、`diff`、`left`、`right`、`max` 为终止读出。图的 `L` 和 `R` 分别给出每个节点的下一节点，规则查询 `has` 输出 0 或 1。每种语言的完整语法和执行语义都在 [tasks.py](ct_rl/tasks.py)。

基础后缀先抽长度，再抽合法操作与最终读出。在训练和 ID 分布中保留单个原语，按任务族各留出一个相邻组合用于 `comp`。训练搜索也遵守同一组合限制。`scale` 只评估图与规则，计算相对 ID 的降幅时使用同一任务族集合。

划分键来自底层对象。算术程序处理寄存器重命名，图按边类型和节点重命名计算规范形式，规则系统用对谓词重命名不变的颜色细化签名分组。签名碰撞会把实例放到同一组。前缀对的两侧、同一对象上的读出与反例进入同一划分。生成的数据记录和 manifest 均保存到本地，读取时校验摘要。

## 对照、消融与计算成本

`--set` 支持修改训练及模型字段。以下命令可在同一份数据上运行 GRPO，或在已有检查点上继续增加训练轮数。

~~~bash
python -m ct_rl train --data runs/demo/data --config configs/tiny.json \
  --out runs/grpo --set algorithm=grpo --set grpo_epochs=2

python -m ct_rl train --data runs/demo/data --config configs/tiny.json \
  --out runs/grpo --set algorithm=grpo --set grpo_epochs=2 --set rounds=5 --resume
~~~

检查点在冷启动完成后及每轮结束后原子写入，包括模型、参考策略、需要恢复的优化器、随机数状态、反例池和历史记录。预训练模型的检查点保存适配器和任务头，加载时还需原始骨干。训练日志记录源码摘要、数据摘要、模型版本、种子与训练配置。

| `method` | 实验条件 |
|---|---|
| `ct-rl` | 多后缀正确率、留一基线、区分性反例课程 |
| `single-readout-rl` | RL 只依据固定基础读出给奖励 |
| `random-suffix-rl` | 多后缀测试从基础分布采样 |
| `random-separating-rl` | 在每个长度桶中随机选取能区分两侧的后缀 |
| `hardness-matched-rl` | 在相同长度内匹配失败率，候选可包含不能区分两侧的后缀 |
| `latent-sft` | 同一潜接口，每次用一个随机后缀做答案监督 |
| `multi-query-sft` | 同一潜接口，对多个后缀做答案监督 |
| `static-compress-sft` | 一次前缀编码生成相同长度和维度的接口，零次额外潜迭代 |
| `placeholder-sft` | 固定长度的全零占位接口，训练续算器 |

三种子比较和按查询次数摊销的成本曲线可分别运行以下命令。

~~~bash
python -m ct_rl.suite --config configs/tiny.json --data runs/demo/data \
  --out runs/comparison --seeds 31 41 51 \
  --methods ct-rl single-readout-rl random-suffix-rl multi-query-sft

python -m ct_rl eval --checkpoint runs/demo/train/checkpoint.pt --data runs/demo/data \
  --out runs/reuse.json --split id --m 1 4 8 16 --serial --count-flops

python -m ct_rl eval --checkpoint runs/demo/train/checkpoint.pt --data runs/demo/data \
  --out runs/interventions.json --split id --m 4 \
  --intervention normal remove zeros shuffle swap prefix-cache
~~~

`remove` 移除接口位置，`zeros` 保留长度并将值归零，`shuffle` 打乱步序，`swap` 在基础答案相同的配对前缀间交换整段 `Z`。`prefix-cache` 是独立诊断条件，刻意把生成侧根据原始前缀建立的 KV 传给续算侧。它的结果需单独报告。

`suite` 默认使用相同训练步数。指定 `--target-flops` 后，它按轮检查含冷启动、反向传播和反例搜索的累计算子 FLOPs，在预算或配置轮数上限处停止。`comparison.json` 给出实际开销与相对偏差。计数器使用 PyTorch 的算子公式，并为 CPU 融合注意力补上同一约定的前向与反向 FLOPs。GPU 峰值显存来自 CUDA 分配器，缓存字节数统计实际 KV 张量。`--serial` 以批量 1 测量延迟，报告前缀开销加上 M 次续算后每个查询的平均耗时。额外配对探针单独计时。

训练时的 `k`、`m`、`rho`、`model.latent_steps` 与 `model.action_dim` 可覆盖。例子为 `--set m=4 --set model.latent_steps=8 --set model.action_dim=64 --set rho=0.25`。

## 如何读评估结果

每个前缀只采样一次潜状态。相同划分、评估种子、M 的不同方法使用相同测试后缀，基础后缀按分布有放回采样。置信区间在任务族内以整对前缀为单位重采样，然后取任务族宏平均。JSON 中的正确率取值为 0 至 1。

- **CT-Acc** 先求每个前缀在 M 个后缀上的平均正确率，再对前缀和任务族汇总。
- **All-Correct@M** 计算同一潜状态答对全部 M 个测试的前缀比例。
- **Pair-Both** 在独立的、符合当前划分约束的区分性后缀上评估一对前缀，两侧同时正确才得分。该探针重用已生成的两份潜状态。
- **invalid_rate** 统计普通测试的格式错误或预算耗尽比例。每次续算预测、配对探针预测、操作数与成本记录均可在 JSON 中检查。

## 公开推理基准

安装公开基准的可选依赖，然后将官方数据置于本地。输入可以是 JSON、JSONL 或 Parquet。GSM8K 从 `####` 后读取标准答案。PrOntoQA 支持官方生成器的 `test_example` 结构及预处理记录。ProsQA 使用 Coconut 发布的数据。MATH-500 使用题目与 LaTeX 答案，并通过 Math-Verify 判断等价性。[GSM8K](https://github.com/openai/grade-school-math) · [PrOntoQA](https://github.com/asaparov/prontoqa) · [ProsQA](https://github.com/facebookresearch/coconut) · [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) · [Math-Verify](https://github.com/huggingface/Math-Verify)

~~~bash
python -m pip install -e '.[public,test]'
python -m ct_rl public-prepare --dataset gsm8k --input official/gsm8k/train.jsonl \
  --out data/gsm8k --valid 500 --seed 120
python -m ct_rl public-adapt --checkpoint runs/qwen15b/checkpoint.pt \
  --dataset gsm8k --input data/gsm8k/train.json --out runs/gsm8k-adapted.pt \
  --mode latent --steps 100 --batch-size 2 --tokens 64
python -m ct_rl public-eval --checkpoint runs/gsm8k-adapted.pt --dataset gsm8k \
  --input official/gsm8k/test.jsonl --out runs/gsm8k-test.json --mode latent --tokens 64
~~~

自然语言题目作为前缀，固定“给出最终答案”的后缀用于评价 Pass@1。`latent` 对 RL 检查点沿用组采样和交替训练，对监督学习控制组沿用答案监督。`direct` 和 `cot` 支持直接答案与显式推理文本的监督和评估。CoT 使用数据中提供的推理步骤，末尾用 `Final answer:` 标记答案。MATH-500 只接受评估。正式实验中应使用官方测试文件，并根据验证集锁定训练配置、算力和解码预算。

## 接入第三方对照方法

草稿列出的 Coconut、CODI、Latent-GRPO、SLPO、LTC 和 RLTT 均有本地训练实现。可在普通训练命令中使用 `--set method=codi` 等选项单独运行。公开原生潜方法通过 `native_pretrain=true` 与 `public-adapt --mode native` 连接。统一潜接口版本及原生公开版本的结构与监督差异见[实验协议](EXPERIMENTS.md)。如果已经持有其他实现的权重，也可以用下面的接口导出相同测试并评分。

~~~bash
python -m ct_rl export-tests --data runs/demo/data --split id --m 4 --out runs/external
python -m ct_rl.external --inputs runs/external/inputs.json \
  --factory my_adapter:load --weights models/my_baseline --out runs/external/pred.json
python -m ct_rl score-external --cases runs/external/scoring.json \
  --predictions runs/external/pred.json --out runs/external/score.json
~~~

`my_adapter:load` 是外部模型的工厂函数，输入权重路径，返回两个方法的对象。`encode(prefix)` 为一个前缀生成可复用的状态，`answer(state, suffix)` 在独立后缀上下文中返回答案文本，预算耗尽可返回 `None`。驱动会先编码两侧前缀，再揭示后缀。统一潜接口实验还需在适配器内部实现对原始前缀文本和 KV 的隔离。

已有预测也可以直接评分。预测文件是以 group 值为键的 JSON 对象，每个值含 `answers` 和 `probe_answers`。前者形状为 `[2][M]`，后者长度为 2。推理输入与评分标签分别位于 `inputs.json` 和 `scoring.json`。
