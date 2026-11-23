# ResponseGuard

本工程依据《ResponseGuard：保持语义可控性与条件多样性的图像生成强化学习》和配套方法问答实现。它从成对提示词构建参考响应，在 Stable Diffusion 3.5 Medium 的随机流采样器上训练 LoRA，并在保留全部生成样本的条件下评测语义响应和多样性。

文稿未提供原始源码、作者检查点和实测超参数。工程使用 BLIP、DINOv2 和 PickScore 实现文中的冻结核验器、视觉编码器和偏好函数，初始阈值及预算可在验证集上校准。主配置固定了生成模型和核验模型的 Hub 修订版本。

## 完整 smoke 与实验入口

完整 smoke 在 CPU 上运行 11 种训练设置、3 个训练种子、分段恢复与外部图片回读，另外用随机初始化的真实 SD3 组件验证 LoRA 训练和导出。报告提供配对图像、训练曲线、汇总统计和运行配置。[本次运行报告](docs/Smoke运行报告.md) · [主实验超参数及选择依据](docs/超参数与实验协议.md)

```bash
python -m pip install -c constraints.txt -e '.[sd3,eval,figures,test]'
python -m scripts.smoke --out runs/my-paper-smoke
```

下面的 demo 适合只运行三种训练设置的快速检查。

## 本地直接运行

在项目目录中使用 Python 3.11。以下命令创建环境并运行离线示例。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints.txt -e '.[test,figures]'
python -m responseguard demo --out runs/demo --figures
```

离线示例使用可训练的低维流策略、几何图形渲染器和模拟器场景标签，覆盖参考缓存、训练、保存与加载、评估和成图。它的距离指标在输出中标为 `tiny_rgb_cosine`，数值用于检查工程流程。

| 运行产物 | 用途 |
|---|---|
| `runs/demo/joint/checkpoint.pt` | LoRA 参数、优化器、历史窗口、乘子和迭代位置 |
| `runs/demo/joint/train.jsonl` | 每次更新的约束、优势方差、裁剪比例、KL、梯度范数和开销 |
| `runs/demo/joint-eval/pairs.jsonl` | 每个提示词、种子、图像、原子核验分数及有效性标记 |
| `runs/demo/comparison/comparison.json` | 共同可评测集合上的指标、覆盖率及配对差异的置信区间 |
| `runs/demo/comparison/paired-seeds.png` | 固定提示词对在各方法、全部种子上的图像网格 |
| `runs/demo/comparison/tradeoff.png`、`training.png` | 有效率与条件多样性、训练约束与计算开销 |

## SD3.5 训练与推理

主配置使用 CUDA 和 BF16。先安装适合机器的 PyTorch 与 torchvision，再安装研究依赖。生成模型权重需要在 Hugging Face 账号下获得访问权限，也可以将配置中的模型名改为本地 Diffusers 模型目录。`device` 和 `judge_device` 可以分别指定设备。单进程训练使用微批次和梯度检查点。[SD3.5 Medium 模型卡](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)

```bash
python -m pip install -c constraints.txt -e '.[sd3,eval,figures,test]'
python -m responseguard data --out data

python -m responseguard train --config configs/sd35.json \
  --data data/train.jsonl --cache cache/refs --out runs/joint --variant joint

python -m responseguard evaluate --config configs/sd35.json \
  --checkpoint runs/joint/checkpoint.pt --data data/test.jsonl --out runs/joint-eval

python -m responseguard generate --config configs/sd35.json \
  --checkpoint runs/joint/checkpoint.pt --prompt "Three red cups on a wooden table." \
  --seeds 7 11 19 --out runs/cups

python -m responseguard export-lora --config configs/sd35.json \
  --checkpoint runs/joint/checkpoint.pt --out runs/joint-lora
```

导出的 `pytorch_lora_weights.safetensors` 可由 `StableDiffusion3Pipeline.load_lora_weights` 加载。内置 `generate` 使用与本工程评测一致的随机采样器。使用原生管线的其他采样器时，生成协议相应改变。[Diffusers SD3 文档](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_3)

```python
pipe.load_lora_weights("runs/joint-lora", weight_name="pytorch_lora_weights.safetensors")
```

首次遇到提示词对时自动构建参考缓存。要在另一时段预生成全部训练参考，可运行下列命令。`--until 10` 可用于先训练到第 10 轮，随后在同一个训练命令末尾添加 `--resume` 继续。断点按原子替换写入，恢复时会校验数据、配置与消融设置。

```bash
python -m responseguard prepare --config configs/sd35.json \
  --data data/train.jsonl --cache cache/refs
```

## 数据与核验器

`data` 默认生成 8,000 对训练提示词、1,000 对验证提示词和 1,000 对测试提示词。每对只修改颜色、数量或空间关系之一。生成器按目标对象、颜色、空间关系三元组分块，成对提示词和局部投影用的单因素替换都留在所属划分。默认种子 1203 产生 512、64、64 种三元组，所有原子概念均在训练中出现。`dataset.json` 记录实际交集计数，`study` 从三个 JSONL 文件重新核验。已有旧版数据时，可用 `python -m responseguard data --out data-v2` 另行生成，并为这批数据指定新的训练输出目录。用 `--sizes 100 20 20` 可减小规模。`data/creative.jsonl` 另含 200 条通用创作提示词及逐条核验问题。

训练用 BLIP 的二选一分数核验对象、参照物、颜色、数量和关系。对任一被生成图像，模型同时回答本提示词与干预后目标属性的问题。所有原子都过阈值才通过语义核验。独立的质量问题给出质量代理分数。每张图的任务奖励由 PickScore 图文余弦相似度、质量分和原子满足度均值加权得到，权重依次为 0.45、0.20 和 0.35。PickScore 单图分数可直接跨组比较。评测换用另一个冻结的 BLIP 检查点，两个核验器属于同一模型家族。正式评测另使用 HPSv2.1 给出独立偏好分。[BLIP VQA](https://huggingface.co/Salesforce/blip-vqa-base) · [PickScore](https://huggingface.co/yuvalkirstain/PickScore_v1)

`audit` 从验证集导出图片与 CSV。在 `semantic_ok` 和 `quality_ok` 中填写人工判断的 0 或 1，再用 `calibrate` 按平衡准确率选择阈值。输出同时给出人工标注数。`--role train` 和 `--role eval` 分别校准训练核验器和评测核验器，将各自的 `sem_threshold`、`quality_threshold` 写入相应配置。预算、SVD 秩和边际容忍度同样可以用验证集选择。

```bash
python -m responseguard audit --config configs/sd35.json \
  --data data/val.jsonl --pairs 20 --role train --out runs/audit-train
python -m responseguard calibrate --labels runs/audit-train/labels.csv \
  --out runs/train-thresholds.json
```

## 方法与代码对应

| 论文步骤 | 实现 | 本工程的具体做法 |
|---|---|---|
| 单因素干预 | `responseguard/data.py` | 生成互斥提示词和原子核验问题 |
| 随机流策略 | `policy.py`、`flow.py` | 初始噪声在两个分支间匹配，后续探索噪声按分支、种子、轮次独立生成 |
| 参考缓存 | `reference.py` | 关闭 LoRA 生成固定参考，两方向均语义正确且间隔达标后按种子存储锚点 |
| 语义响应信赖域 | `objective.py` | 对两方向分别计算参考间隔减去当前间隔的单侧平方惩罚，按 `K` 归一 |
| 局部语义子空间 | `reference.py`、`objective.py` | 在每个提示词附近对五种显式属性做干预，对至少两组核验通过的匹配种子取平均特征差，做截断 SVD |
| 条件多样性约束 | `objective.py` | 在正交补上计算合格样本两两平方距离。参考或当前有效样本少于两个、或缺少可靠投影方向时，仅保留有效率惩罚 |
| 组级优势与裁剪 | `train.py`、`flow.py` | 完整生成组共用一个优势，逐分支逐去噪步裁剪。历史基线、尺度、乘子在同一批轨迹的 PPO 更新中固定 |
| 评价与比较 | `evaluate.py`、`compare.py` | 输出 PairSR、保留率、错误修正率、未修改语义通过率、有效率、条件和有效成对多样性、Vendi 及覆盖率 |

训练侧视觉特征为归一化的 DINOv2 CLS 向量。局部基由参考干预确定。当前策略的变化只影响当前响应和当前样本的残余特征。语义锚点与跨种子多样性分别核验，训练日志保存各自的覆盖信息。

主配置的 `k_train=4` 表示每个分支 4 张图，每组共 8 张。`k_ref=16` 用于匹配参考锚点和多样性下界。另取 `k_cal=16` 个独立初始种子估计参考有效率。每个提示词的有效率下界为 `max(rho_floor, cal_rate - rho_slack)`，默认 `rho_floor=0.5`、`rho_slack=0.1`。测试每个提示词生成 `k_eval=16` 张。

历史窗口保存最近 256 个完整组的原始收益和代价。每批先用当前乘子重算历史收益，优先减去同一提示词对的历史均值，样本缺失时回退到全局历史均值。冷启动用基线 0、尺度 1，此后用全局历史标准差，最小值为 0.1。对同组图像共享的优势直接进入裁剪目标。每批新轨迹只更新一次乘子，更新式为 `max(0, lambda + dual_lr * (mean_cost - budget))`。

ResponseGuard 的逐步比率使用整个潜变量转移的高斯对数密度。实现通过新旧均值之差计算对数比率，避免两个很大的对数密度直接相减。`clip_eps=0.1` 作用于每个随机转移。`flow-grpo` 对照采用公开实现的逐坐标平均对数密度、按提示词标准化的单图收益、冻结参考高斯 KL，并单独配置裁剪范围与 KL 权重。比较时使用本工程相同的数据和奖励。[Flow-GRPO 论文](https://arxiv.org/html/2505.05470v5) · [公开采样实现](https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py)

## 消融、评测与绘图

`--variant` 支持 `joint`、`semantic`、`diversity`、`task-only`、`unprojected`、`per-prompt`、`flow-grpo`、`diversegrpo`、`drift`、`margin-supervised` 和 `pair-grpo-adapted`。前四个构成两个约束的完整因子设计。`unprojected` 同时改用未投影的当前距离与参考下界，保持参考覆盖集合一致。`per-prompt` 是相同成对数据上的单图优势对照。`--no-history` 关闭历史基线和尺度缩放。

```bash
python -m responseguard evaluate --config configs/sd35.json \
  --data data/test.jsonl --out runs/base-eval
python -m responseguard compare --runs runs/base-eval runs/joint-eval \
  --out runs/compare --bootstrap 1000 --figures --pair-index 0
python -m responseguard curves --runs runs/joint --out runs/compare/training.png
python -m responseguard creative --config configs/sd35.json \
  --data data/creative.jsonl --checkpoint runs/joint/checkpoint.pt --out runs/creative
```

`d_cond` 使用通过语义与质量核验的图像在 DreamSim ensemble 空间中的平均两两距离。`d_eff` 将同一距离之和除以全部可能的图像对数。合格样本少于两个时，前者写为 JSON `null`，后者为 0。`cov2` 为至少有两个合格样本的提示词比例。`vendi` 使用固定数量合格样本的归一化嵌入 Gram 矩阵，`cov_m` 报告对应覆盖率。正式评测还输出独立 DINOv2-Base 空间的辅助距离 `aux_d_cond`，以及有效图像对比例 `valid_pair_fraction`。[DreamSim 官方实现](https://github.com/ssundaram21/dreamsim) · [Vendi Score 定义](https://arxiv.org/abs/2210.02410)

`compare` 对条件多样性和 Vendi 使用所有参与比较的方法共同可计算的提示词集合。它以完整提示词对为重采样单位，保存相对首个方法的配对 bootstrap 区间。图中的误差线表示测试提示词抽样的不确定性。不同训练种子之间的均值和标准差由 `aggregate` 单独计算。比较结果还将 D_eff 的差值分解成有效图像对比例、共同有定义时的距离变化及共同集合外贡献。对 DiverseGRPO 和 DRIFT 另报告语义与质量通过率各相差至多一张图时的条件距离差。

运行完整的三个训练种子与全部对照可执行 `bash scripts/run_experiments.sh configs/sd35.json runs/study data cache/refs`。该脚本会执行 33 次主训练，保存验证里程碑及测试结果，并按实际去噪开销、核验量或设备小时匹配预算。单次试验可直接使用上面的 `train` 命令。每种方法都记录实际生成、去噪重放、额外参考前向、语义核验与偏好评价次数。每个用到的参考缓存计费一次，包含预计算的缓存。`charged_device_hours` 是在线运行和已计入参考构建的墙钟时间乘以分配的 CUDA 设备数。

`validation-budget.json` 的 `points` 保存冻结起点和全部实测值。某个上限进入 `matched` 的条件是每种方法都有至少一次更新后、费用不超上限的实测检查点。`coverage` 记录上限、各方法首次训练成本和可比较点数。预算图按选中的检查点绘制阶梯曲线，其中的条件距离采用该预算下各方法共同可计算的验证提示词。

主结果中的 `d_cond_common` 和 `vendi_common` 分别使用全部训练方法、训练种子和冻结参考的提示词交集，先对同一批提示词求均值，再计算训练种子间的均值与样本标准差。`common_support` 给出各指标的分子、分母、覆盖率和提示词集合指纹。空交集的指标写为 `null`。各方法自身覆盖上的 `d_cond`、`vendi` 一并保留。`aggregate` 校验训练轮数、训练样本指纹、超参数及历史基线设置。单独复算某种方法时，可用 `--common-runs` 加入其他方法的同批种子和冻结参考。

外部方法可以按下列清单生成相同提示词与初始种子的图片。清单中的 `image` 相对于清单所在目录。用 `evaluate --external` 读入这些图片后，语义判定、参考保留率与多样性走相同的评测代码。

```bash
python -m responseguard eval-plan --config configs/sd35.json \
  --data data/test.jsonl --out runs/external/manifest.jsonl
python -m responseguard evaluate --config configs/sd35.json --data data/test.jsonl \
  --external runs/external/manifest.jsonl --name ExternalMethod --out runs/external-eval
```

## 公开基准

GenEval 2 导出官方评测器要求的“提示词到图像绝对路径”字典。生成后，在 GenEval 2 自己的 Python 环境运行 Soft-TIFA。将命令中的数据和仓库路径指向本地的官方发布版本。[GenEval 2 官方数据与评测](https://github.com/facebookresearch/GenEval2)

Soft-TIFA AM 先对每条提示词的原子分数取算术平均，再对提示词等权平均。GM 在第一步取几何平均，总分同样对提示词等权。结果文件使用 0 到 1 的量纲，乘以 100 对应官方百分制。配对 bootstrap 以提示词为抽样单位。已有逐原子分数和生成清单的完整公开评测可用 `python -m scripts.public_eval --scores-only --study runs/study --geneval-repo upstream/GenEval2 --out runs/public` 重算汇总及区间。[官方评分实现](https://github.com/facebookresearch/GenEval2/blob/main/evaluation.py#L173-L178)

```bash
python -m responseguard benchmark --format geneval2 --config configs/sd35.json \
  --data upstream/GenEval2/geneval2_data.jsonl --checkpoint runs/joint/checkpoint.pt --out runs/geneval2
python upstream/GenEval2/evaluation.py --benchmark_data upstream/GenEval2/geneval2_data.jsonl \
  --image_filepath_data runs/geneval2/image_paths.json --method soft_tifa_gm --output_file runs/geneval2/scores.json
python upstream/GenEval2/soft_tifa_analysis.py --benchmark_data upstream/GenEval2/geneval2_data.jsonl \
  --score_data runs/geneval2/scores.json
```

DeltaBench 按每对提示词、每个方向、每个种子组织四张图。`--format deltabench` 读取用户提供的基准 JSON 数组，每项包括 `category`、`caption1`、`caption2`。`questions1` 和 `questions2` 保存对应方向的完整全局与细粒度核验问题。每个问题可以是预期回答为 yes 的字符串，或含 `question` 与 `answer` 的对象。`scripts/score_deltabench.py` 是依论文公式编写的评测程序，读取这些问题，用 Qwen3-VL-30B-A3B-Instruct 打分后输出分任务的算术均值、按提示词对的几何均值和类别宏平均。实际报告需记录提示词版本、问题清单和评测模型修订版。[DeltaBench 论文](https://proceedings.mlr.press/v306/pan26b.html) · [Qwen3-VL 检查点](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)

```bash
python -m responseguard benchmark --format deltabench --config configs/sd35.json \
  --data upstream/deltabench.json --checkpoint runs/joint/checkpoint.pt --out runs/deltabench
# 在含 torch、Pillow、accelerate、transformers==4.57.6 的独立评测环境中执行
python scripts/score_deltabench.py --manifest runs/deltabench/images.jsonl --out runs/deltabench/scores
```

作者公开仓库另有名为 PairComp 的发行版。使用该发行版时，选择 `--format paircomp`、输入 `PairComp/prompt/PairComp.json`，再用该版本提供的 `evaluate_images.py` 和 `summary_scores.py` 评价。输出文件名遵循 `X_Y_Z.png`，实验表中应保留实际基准版本名。[PairComp 官方说明](https://github.com/wendell0218/FocusDiff/tree/main/PairComp)

## 验证范围

`python -m pytest -q` 可运行方法与集成测试。安装 `sd3` 扩展后，集成测试会在本地创建随机初始化的小型 SD3、VAE、文本编码器、BLIP、DINOv2 和 CLIP 检查点，实际运行参考生成、LoRA 反向更新、检查点重载、Diffusers 原生 LoRA 导入及基准图像导出。

本地完成的是 CPU 流程与小型 SD3 组件验证。正式 SD3.5 Medium 权重的训练、DreamSim 预训练权重评测、公开基准全量打分与人工阈值校准，均需要相应模型、算力和标注。论文中的收益数值应取自那些正式运行的输出。
