# 10% 标注：监督继续训练与 BCP 区域混合验证

更新：2026-09-14。MambaLiteUNet 已由用户放弃；本轮不启动它，也不继续扩展旧扫描半监督配方。当前任务是复用已有 10% 标注监督权重，验证未标注区域混合能否产生额外收益。

本文只定义当前已实现的一轮实验。后续机制对照、训练种子43/44、进入条件与租赁时间统一见 [GPU后续执行计划](GPU后续执行计划.md)。后续实现尚未包含在 `run_bcp10.sh` 中。

## 当前证据与方法定位

189 张标注的监督模型最佳验证 pooled Dice=0.8698412827556765，pooled IoU=0.7696629415703827，best epoch35；同标注图的旧扫描 SSL Dice=0.8556532181003861。原始日志核查见 [结果记录](SSL结果核对与下一步_2026-09-13.md)。不同训练配方和更新次数限制了因果解释，因此本轮使用同一个初始权重、同一标注采样和更新预算。

参考 [BCP（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Bai_Bidirectional_Copy-Paste_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.html) 与[官方实现](https://github.com/DeepMed-Lab-ECNU/BCP)。本仓库实现是针对二分类 VM-UNet 的机制适配，不是论文原样复现，更不是原创贡献声明。原论文的主要实验不是 ISIC，不能承诺它在当前数据上的提升。

## 固定的两组实验

| 设置 | 纯监督继续训练 | BCP 区域混合适配 |
|---|---|---|
| 初始化 | 现有10%监督 best | 完全相同的 best；EMA 教师也由此初始化 |
| 骨干 | 原 VM-UNet | 原 VM-UNet，无新增结构 |
| 标注数据 | 固定189张 | 相同189张 |
| 未标注数据 | 不参与训练 | 固定1697张，仅读取图像 |
| 每轮标注采样 | batch32，最后一批29；每张一次 | 相同顺序和几何增强抽样，每张出现在一对互补混合图中 |
| 更新预算 | 300轮×6步=1800次 | 300轮×6步=1800次 |
| 优化器 | 新建 AdamW，lr=1e-4、wd=.01 | 相同 |
| 学习率 | cosine T_max300，eta_min=1e-5 | 相同 |
| 输入与标签 | 256，corrected、最近邻掩码 | 相同 |
| 验证 | 每轮808张，阈值.5，最高pooled Dice选模 | 相同，只评估学生 |
| 初始模型保留 | epoch0作为初始best | 相同；没超过初始值不算增益 |

新阶段重置优化器，不加载原 epoch35 的优化器状态；不重新花GPU做监督预训练。300轮在这里仅1800次更新，旧10% SSL是31800次；本轮是预先限定预算的机制筛选，失败不等于否定所有预算下的BCP。

两组优化更新数和标注图抽样匹配，计算成本并不相同。候选每步一个教师前向、两个混合学生前后向；两个学生图依次反传再统一更新，避免同时保留两份激活。GT区域在两幅图中互补覆盖，但它们处于不同上下文，损失计算也有区域归一化；不能把结果差异简单解释成只有“更多数据”。若候选正向，后续需要有标注图之间混合的对照，区分增强与未标注学习的作用。

### 候选的具体机制

1. 从固定1697张图像中按训练步确定随机采样及增强；不读取真实掩码，也不使用其标签做质量筛选。完整标注目录里的隐藏GT不应进入训练路径。
2. EMA教师在eval/no-grad模式预测增强后的未标注图，阈值.5生成二值伪标签。
3. 随机裁出高、宽各约2/3的矩形区域，用同一个矩形构造“标注外部+未标注内部”和互补图；GT、伪标签的位置与图像一致。
4. 两个区域分别计算BCE+soft Dice；GT权重1，伪标签权重.5，按总权重归一化后对双向损失取均值。矩形和权重是固定适配选择，未经本项目最优性验证。
5. 每次学生优化后以EMA=.99更新教师；保持四向扫描完整，不使用旧方向一致性、扫描丢弃或R4教师约束。

适配差异：使用现成VM-UNet少标注权重，而非BCP原程序的ACDC预训练；使用二分类概率/BCE、当前AdamW和增强；不移植ACDC多类别连通分量后处理，也不添加未经验证的置信度筛选。未标注采样的Python随机数与标注增强隔离，按seed/epoch/batch确定；EMA、优化器、调度器与原RNG均保存，支持轮次边界精确恢复。

## GPU 执行

服务器之前有大量未提交改动，推荐由监控智能体保留旧工作区，建立独立目录。先确认没有同名正在运行的任务；以下命令不会更改原实验目录的源码：

```bash
git clone --branch main https://github.com/JWJW000/VM-UNet.git /home/linux/work/VM-UNet-bcp10
cd /home/linux/work/VM-UNet-bcp10
mkdir -p data results splits
ln -s /home/linux/data/isic2018 data/isic2018
ln -s /home/linux/work/VM-UNet/results/full_fewlabel_r0p1_s42 results/full_fewlabel_r0p1_s42
cp /home/linux/work/VM-UNet/splits/isic18_r0p1_s42.json splits/isic18_r0p1_s42.json
conda activate vmunet
nohup bash run_bcp10.sh > bcp10.log 2>&1 &
tail -f bcp10.log
```

目录/链接已存在时先检查，不重复clone、不删除重来。如果原服务器产物路径改变，用保留的 `ssl_clean_pull/fewlabel_r0p1_s42` 内容恢复为 `results/full_fewlabel_r0p1_s42`，其中需包含best.pth和manifest.json；SSL split原文件恢复到splits目录，不重新生成划分。脚本检查以下指纹：

- best.pth：`05e429b4f8c5dfed729fe3068fbb0c9a5a827de8fd1c76956acca7f088e8d1fc`
- 监督manifest：`d1d4f587a50546f114caa158ca8261f89b43f93f16623598a7fa69df7707e0dd`
- SSL split：`621d499b3cd6527c046ff99aaf84e11f761797f9f7aae4236bc98523d2d74c00`

脚本顺序：输入指纹与CUDA检查 → BCP独立目录的一轮六步烟测 → 学生/EMA权重重载及CUDA推理 → 纯监督组1800步和最佳权重评估 → BCP组1800步和评估 → CPU汇总。任一步失败即停止后续任务；烟测权重不用于正式初始化。

每组训练前重评初始权重，Dice与0.8698412827556765差超过1e-6就停止，保留initial_metrics.json排查代码/数据。候选烟测同时验证真实Mamba内核、前后向与显存；本机CPU测试不能代替它。不要通过放宽门槛、改batch、换初始化来绕过错误。

输出：

- `results/bcp10_smoke_s42`：工程检查，不计入方法结果。
- `results/bcp10_control_s42`、`results/bcp10_method_s42`：正式两组。
- `results/diagnosis_bcp10_control_s42`、`results/diagnosis_bcp10_method_s42`：逐图与边界诊断。
- `results/bcp10_comparison.json`：两组最佳/末轮、耗时与筛选结果。

每组保存config、初始化/划分/关键源码指纹、实际关键源码压缩包、环境、initial/best/latest、metrics.csv。表内`optimizer_updates`核对步数，`peak_cuda_memory_mb`记录峰值分配显存，seconds含本轮训练与验证；归档与单独分析耗时另计。两组训练损失定义不同，不直接比较它们的train_loss大小。

### 中断与恢复

不重复运行整条脚本。先辨认当前进程与父脚本、检查CSV/latest。仅对意外中断且latest有效的同一阶段恢复；已经完整结束的阶段直接复用结果。示例为候选组，恢复对照组时删掉`--bcp`并改output：

```bash
python -u train_full.py --model vmunet --data-path data/isic2018 \
  --manifest results/full_fewlabel_r0p1_s42/manifest.json \
  --init-checkpoint results/full_fewlabel_r0p1_s42/best.pth \
  --unlabeled-split splits/isic18_r0p1_s42.json --expected-initial-dice 0.8698412827556765 \
  --seed 42 --size 256 --batch-size 32 --lr 0.0001 --weight-decay 0.01 \
  --eta-min 0.00001 --t-max 300 --preprocessing corrected --gpu 0 \
  --bcp --epochs 300 --output results/bcp10_method_s42 --resume
```

源码或配置指纹改变会拒绝恢复。烟测、哈希、起始Dice或OOM失败都先修根因，不能无限原样重试。恢复完成后只补尚未完成的评估/候选阶段；从`run_bcp10.sh`取对应命令，不让整个脚本重跑已完成组。

## 完成后的判断与边界

第一轮只筛选。候选必须满足：Dice ≥0.8798412827556765（相对初始至少+1个百分点），同时Dice/pooled IoU均超过匹配纯监督继续训练对照，才值得扩大。该1个百分点是预先设定的投入目标，不是效果承诺或显著性标准。pooled Dice与IoU单调对应，不算两份独立证据；还需看macro Dice、边界与严重失败病例。

- 达标：汇报全部结果，按 [GPU后续执行计划](GPU后续执行计划.md) 的阶段2准备“有标注图混合”机制对照，通过后才进入阶段3的多种子验证。后续方案已写明，代码仍需补齐；不自动扩展5%/20%比例或启动多个新模块。
- 小幅正向但未达目标：记录收益与成本，暂停扩大；用已导出的失败图形成下一份具体方案，不能直接改权重重跑。
- 无收益/退化：停止这一个候选，继续CPU证据分析，区分当前预算/适配局限与机制问题；不得据此宣称BCP论文错误。
- 工程阻断：记录具体错误与已耗资源，修复兼容问题后再决定恢复，禁止掩盖协议变化。

189张监督best曾按这808张验证图选择，后续全部结果仍为开发验证。只有形成独立贡献、同协议强对照、多种子与独立/外部数据证据，才可发展为论文结论。代码接入不等于机制有效。

## 本机验证记录

2026-09-13：39项测试通过。包括二值区域损失、双向分次反传与联合反传梯度一致、教师无梯度、未标注GT文件缺失时仍可训练、划分重叠/路径/字节重复拒绝、初始Dice不匹配时训练前停止，以及纯监督和BCP的学生/EMA断点恢复一致。既有模型/评估测试保留通过。

真实189张监督best在当前VM-UNet严格加载成功，参数27,427,561，checkpoint epoch35，记录的Dice=0.8698412827556765。该检查只证明参数兼容，未在本机进行真实CUDA推理；服务器初始重评及烟测仍为正式训练的必要启动条件。
