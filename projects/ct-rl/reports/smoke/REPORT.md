# CT-RL 实验记录

运行配置为 smoke。主比较使用种子 31、41、51。

| 受控实验 | 已测试的独立训练运行 |
|---|---:|
| primary | 57 |
| scale-confirmation | 12 |
| matched | 57 |
| ablations | 20 |
| capacity-matched | 12 |

## 主方法的受控测试

| 划分 | CT-Acc 均值 | 跨种子 SD | All-Correct | Pair-Both |
|---|---:|---:|---:|---:|
| id | 0.00% | 0.00% | 0.00% | 0.00% |
| comp | 0.00% | 0.00% | 0.00% | 0.00% |
| prefix_length | 0.00% | 0.00% | 0.00% | 0.00% |
| suffix_length | 0.00% | 0.00% | 0.00% | 0.00% |
| scale | 0.00% | 0.00% | 0.00% | 0.00% |

## 统一潜接口与等算力比较

| 方法 | ID CT-Acc | All-Correct@4 | Pair-Both | Comp-OOD CT-Acc |
|---|---:|---:|---:|---:|
| ct-rl | 0.00% | 0.00% | 0.00% | 0.00% |
| random-suffix-rl | 0.00% | 0.00% | 0.00% | 0.00% |
| single-readout-rl | 0.00% | 0.00% | 0.00% | 0.00% |
| hardness-matched-rl | 0.00% | 0.00% | 0.00% | 0.00% |
| random-separating-rl | 0.00% | 0.00% | 0.00% | 0.00% |
| latent-sft | 13.89% | 0.00% | 0.00% | 30.56% |
| multi-query-sft | 13.89% | 0.00% | 0.00% | 22.22% |
| static-compress-sft | 16.67% | 0.00% | 0.00% | 19.44% |
| placeholder-sft | 15.28% | 0.00% | 0.00% | 12.50% |
| coconut† | 0.00% | 0.00% | 0.00% | 0.00% |
| codi† | 8.33% | 0.00% | 0.00% | 18.06% |
| latent-grpo† | 6.94% | 0.00% | 0.00% | 18.06% |
| slpo† | 6.94% | 0.00% | 0.00% | 18.06% |
| ltc† | 6.94% | 0.00% | 0.00% | 18.06% |

## 循环骨干比较

| 方法 | ID CT-Acc | All-Correct@4 | Pair-Both | Comp-OOD CT-Acc |
|---|---:|---:|---:|---:|
| loop-grpo | 6.94% | 0.00% | 0.00% | 18.06% |
| rltt | 6.94% | 0.00% | 0.00% | 18.06% |
| loop-ct-rl | 11.11% | 0.00% | 0.00% | 30.56% |

## 可读取原始前缀的文本参照

| 方法 | ID CT-Acc | All-Correct@4 | Pair-Both | Comp-OOD CT-Acc |
|---|---:|---:|---:|---:|
| direct | 0.00% | 0.00% | 0.00% | 0.00% |
| cot | 0.00% | 0.00% | 0.00% | 0.00% |

按验证集规则入选第二规模的三个方法为 placeholder-sft, multi-query-sft, latent-grpo。分数相同时依次使用 Pair-Both、All-Correct 和方法名。

按训练 FLOPs 匹配的实验共 62 个，最大绝对偏差 6.94%。固定后缀读取次数的每个消融配置完成 64 次策略候选续算。

## 公开任务

共完成 69 次独立任务适配、161 次验证或测试。下表列出主方法在首个种子的结果。MATH-500 使用 GSM8K 适配权重。

| 数据集 | 测试题数 | Pass@1 |
|---|---:|---:|
| gsm8k | 3 | 0.00% |
| math500 | 3 | 0.00% |
| prontoqa | 3 | 0.00% |
| prosqa | 3 | 0.00% |

| 适配任务 | 验证集最高 Pass@1 | 同分第一的方法数 | 入选零合成对照的方法 |
|---|---:|---:|---|
| gsm8k | 0.00% | 15 | coconut-native |
| prontoqa | 0.00% | 15 | coconut-native |
| prosqa | 0.00% | 15 | coconut-native |

验证分数并列时，按方法名选择一个零合成对照。

公开适配阶段相对各数据集共同 FLOP 目标的最大绝对偏差为 10.00%。预算包含公开训练划分上的冷启动和后续学习。三档推理预算由验证集的实测成本确定。

[公开结果均值与标准差](public-aggregate.json) · [训练成本与执行器调用](training-costs.json) · [推理预算与测试实际开销](inference-budgets.json) · [容量消融的等算力结果](capacity-budget.json)

这次在 CPU 上使用随机初始化的小型 Qwen2。第一规模为一层 32 维，第二规模为两层 48 维。每个 CT 测试划分每族取一对前缀，公开任务每个测试集取三题。这些数值验证代码路径和计分协议，正式准确率需用完整数据及 Qwen 权重重新测量。

[全部受控指标](controlled-results.json) · [公开基准逐项得分](public-results.json) · [FLOP 预算](budget.json) · [消融配置](ablation-index.json) · [验证选模记录](selection.json)
