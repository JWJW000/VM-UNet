# MambaLiteUNet 完成后的 GPU 执行计划

更新：2026-09-09。供当前负责监控 GPU 的智能体执行。本文是本轮后续任务顺序的执行依据；研究背景和固定协议见 [研究路线第 5.3 节](后续研究路线.md)。

目标：首轮有足够收益才扩大验证，复用已有结果，避免重复训练和空转。本次只提交计划，不创建新的监控任务、不连接或操作 GPU。当前监控智能体可按以下明确门槛依次开展 seed42、44；不需要在每个已满足的阶段再次询问。最多新增两个完整训练，不自动扩展模型或超参数搜索。

## 1. 接管时先辨认当前状态

- 查看现有训练进程、父 shell、日志与输出目录，确认只有一个执行者负责续跑。不要重复执行 `run_mambalite.sh`：它会重新进入烟测，且正式目录已存在时会退出。
- 既有脚本已安排 seed43 训练结束后自动执行 `analyze_full.py`。父脚本仍在运行时等待它完成，不并行启动第二份评估或新训练。
- 训练正常时继续原定 300 轮，不按早期 Dice 提前终止，不因短时日志停顿启动第二份任务。监控读取日志/CSV即可，不反复加载模型跑验证。
- 当前运行期间不更新训练源码或环境。文档可单独读取；需要同步代码时，等待任务结束并先保留服务器本地修改。不要为了拉取本文停止正常训练。
- 进程已退出但不足 300 轮，先查错误。可恢复的中断从同目录 `latest.pth`、相同代码/源码 SHA256、相同参数加 `--resume` 恢复；不要从头再跑。遇到非有限数值、OOM、配置或依赖不匹配，保存证据后停止自动队列，不反复原样重试或静默改 batch。

## 2. seed43 完成后：先用 CPU 判定是否值得继续

完成证据必须同时满足：正式目录 `results/full_mambalite_s43` 的 `metrics.csv` 连续覆盖 epoch 1–300、`latest.pth` 的 epoch=300，且训练进程已退出。烟测目录的一轮结果不算正式结果。CPU 检查检查点时使用 `map_location='cpu'`，无需占用 CUDA。

核对正式 `config.json`：model=mambalite、seed43、size256、batch32、epochs300、T_max300、lr0.001、weight_decay0.01、eta_min0.00001、corrected、无教师/边界加权/VM-UNet 变体。清单 SHA256 必须是：

```text
5e5790bc5a5f8410c908b4ae19683f03b041d94110199a8c4ed2ecb860707238
```

核对 `best.pth` 中 epoch/metrics 与 CSV 中最高 pooled Dice 对应；若最佳值并列，按训练器实际保留的第一次最高值对应的行比较。若已有 `results/diagnosis_full_mambalite_s43/summary.json`，核对其 checkpoint、图像清单和 pooled 指标；同口径重评允许不超过 1e-6 的绝对浮点误差，超出则排查，不能直接比较。若评估尚未执行，仅补一次最佳权重评估，不重训。

优先从 `results/full_b0_s43/best.pth` 读取基线指标与配置并确认协议；既有记录为 Dice=0.8996483455035698、IoU=0.8176007568374029。指标使用 0–1 原值判断，不对四舍五入后的百分数判门槛。

| seed43 结果 | 立即行动 |
|---|---|
| Dice 相对匹配 B0 至少 +0.005，且 pooled IoU 提高 | 确认评估和归档完成后，进入 seed42 |
| Dice 有提高但小于 +0.005 | 记录“小幅收益，未达投入门槛”，停止新增 GPU 任务并通知用户 |
| Dice 不提高 | 记录“当前固定协议下未改善”，停止新增 GPU 任务并通知用户 |
| 结果不完整、协议不符、重评不一致 | 暂停自动队列，先处理工程问题，不据此判定方法成败 |

按现有 B0 原值，首轮门槛为 Dice ≥0.9046483455035698（约 90.46483455%）。这只是预先约定的资源投入门槛，不是统计显著性或论文录用标准。pooled Dice/IoU 单调对应，不当成两份独立证据；macro Dice、边界与严重失败病例仍必须如实记录。

## 3. 首轮达标后：串行运行 seed42，再决定 seed44

启动前保存 seed43 的配置、environment、CSV、best/latest、日志与诊断结果，记录下一步决定。确认 GPU 上没有当前父脚本/评估或其他用户任务；不杀死不属于本实验的进程。相同模型、数据与环境已通过烟测，无需为每个 seed 重复一轮烟测。

所有设置沿用 seed43，唯一变化为 seed 和输出目录。从官方随机初始化开始；不使用 seed43 的 best 初始化其他种子，也不加载 B0 权重。已有 B0 seed42/44 直接复用，核对配置后无需重训。若相应 B0 结果缺失或协议不匹配，先报告缺口，不在无人知情时扩大训练预算。

先执行 seed42（在仓库根目录、原 `vmunet` 环境中）：

```bash
python -u train_full.py --model mambalite --data-path data/isic2018 \
  --manifest splits/full_isic18_legacy.json --output results/full_mambalite_s42 \
  --seed 42 --size 256 --batch-size 32 --epochs 300 --t-max 300 \
  --lr 0.001 --weight-decay 0.01 --eta-min 0.00001 \
  --preprocessing corrected --gpu 0

python -u analyze_full.py --ckpt results/full_mambalite_s42/best.pth \
  --data-path data/isic2018 --manifest splits/full_isic18_legacy.json \
  --output results/diagnosis_full_mambalite_s42 --gpu 0
```

上面两个命令按成功状态依次执行；训练失败时不得接着评估一个中间 best 并当成完整结果。由当前监控智能体保持任务托管与日志，不另起无人管理的重复队列。每个已存在且完整的输出只读取一次；中断目录按原命令加 `--resume`，不删除后重建。

seed42 完成后按第 2 节核对完整性，比较同种子 B0（既有 Dice=0.8888527176874913）：

- Dice 和 pooled IoU 均正向：继续 seed44。seed42 不单独要求 +0.5 个百分点；最终检查三种子平均提升。
- Dice 或 pooled IoU 不正向：停止 seed44，报告“已不满足每个种子正向的预设标准”。这是为节约预算停止后续扩展，不是提前中止 seed42；不得将两个种子包装成完整三种子验证，也不得隐去失败结果。

只有通过上述条件才执行 seed44：

```bash
python -u train_full.py --model mambalite --data-path data/isic2018 \
  --manifest splits/full_isic18_legacy.json --output results/full_mambalite_s44 \
  --seed 44 --size 256 --batch-size 32 --epochs 300 --t-max 300 \
  --lr 0.001 --weight-decay 0.01 --eta-min 0.00001 \
  --preprocessing corrected --gpu 0

python -u analyze_full.py --ckpt results/full_mambalite_s44/best.pth \
  --data-path data/isic2018 --manifest splits/full_isic18_legacy.json \
  --output results/diagnosis_full_mambalite_s44 --gpu 0
```

seed44 的既有 B0 Dice=0.8974957586777578。所有基线 IoU、macro Dice及附加指标均从对应历史产物读取，不用其他种子或其他模型的值代替。

## 4. 三种子后的明确出口

完成 seed42/43/44 后，以配对方式汇总 B0 与 MambaLiteUNet 的 best 验证成绩：

- 每个种子的 Dice/IoU、差值、最佳轮次、最后一轮成绩、macro Dice、boundary F1、HD95 和未定义数量。
- 三种子均值、样本标准差（ddof=1）、每个种子的配对差值。百分比与百分点分开；不只挑有利种子或轮次。
- 各轮耗时、总训练/评估时间、峰值 CUDA 分配显存。烟测成本单列，不隐藏在速度比较中。
- 清单指纹、模型源码指纹、代码版本、环境、随机初始化与 B0 预训练的差异。该实验比较完整方案，不能把差异全部归因于某个单独模块。

**进入论文方法开发的门槛：三种子平均 Dice 至少 +0.005、每个种子 Dice 均正向、平均 pooled IoU 提高。** 同时如实说明 macro/边界或失败病例的退化，不能称为全面改善。没有满足上述条件就暂停该候选，不自动改学习率、堆模块、增加轮数或转跑 EMCAD。

若通过：下一阶段先做不占训练 GPU 的工作——结合已导出的失败图提出具体问题、检查已有方法是否已经解决、确定一个核心改动及其消融设计。MambaLiteUNet 为已有方法，必须作为直接对照；不能仅凭替换 VM-UNet 宣称原创。第二数据集、独立测试与新结构训练在方案明确后再安排，不自动下载数据或排队未知成本实验。

## 5. 交付记录与资源收尾

每完成一个种子或发生阻断，在 `results/mambalite_followup_status.md` 记录：完成/失败时间、实际进程状态、目录、版本与指纹、指标原值、门槛判定、已运行的下一步及原因。汇总无需 GPU；不要为了生成报告重复训练或反复推理。用户应在首轮判定、失败/阻断和最终汇总时收到简短更新，正常训练期间不持续发送无变化的消息。

达到停止条件或三个种子全部完成后，不再启动额外任务。确认日志与 best/latest 等产物已持久化，按用户已有租赁平台授权处理闲置计费；若没有实例暂停/释放授权，及时提醒用户处理，不擅自销毁实例或数据。停止 Python 不等于停止计费。

本计划不修改当前训练代码，也不会自动驱动另一智能体。负责监控的智能体需要读取并执行本文；GitHub 推送成功不代表它已经接收或执行。
