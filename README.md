# VM-UNet
This is the official code repository for "VM-UNet: Vision Mamba UNet for Medical
Image Segmentation". {[Arxiv Paper](https://arxiv.org/abs/2402.02491)}

## 当前实验：10%标注的 BCP 区域混合验证

MambaLiteUNet 已放弃，旧扫描半监督队列停止扩展。当前保留VM-UNet，复用Dice=86.9841%的少标注监督权重，对比纯监督继续训练与BCP双向区域混合。

[本轮方案与部署命令](docs/BCP少标注验证方案.md) · [后续实验与租赁预算](docs/GPU后续执行计划.md) · [研究路线与历史结论](docs/后续研究路线.md) · [结果核对](docs/SSL结果核对与下一步_2026-09-13.md)

准备好固定权重、manifest和SSL split后，在原CUDA/Mamba环境执行：

```bash
conda activate vmunet
nohup bash run_bcp10.sh > bcp10.log 2>&1 &
tail -f bcp10.log
```

服务器存在未提交代码时，按方案建立独立目录，不直接覆盖。脚本先检查指纹、复现初始Dice及CUDA烟测，再分别运行两组1800次更新，最后评估并汇总；任何阶段失败就停止后续阶段。39项CPU测试通过，尚未在GPU证明适配效果。

2026-09-14 后续安排：本轮达标后先做有标注图混合对照，再根据结果补训练种子43、44。RTX 3090 24GB可先预留12小时；各阶段均满足条件时，累计训练预算约24～32小时。后续两阶段仍需实现、检查和独立启动，当前脚本不会自动执行；准备代码、等待和重新配置环境的时间不包含在训练估算中。

## Abstract
In the realm of medical image segmentation, both CNN-based and Transformer-based models have been extensively explored. However, CNNs exhibit limitations in long-range modeling capabilities, whereas Transformers are hampered by their quadratic computational complexity. Recently, State Space Models (SSMs), exemplified by Mamba, have emerged as a promising approach. They not only excel in modeling long-range interactions but also maintain a linear computational complexity. In this paper, leveraging state space models, we propose a U-shape architecture model for medical image segmentation, named Vision Mamba UNet (VM-UNet). Specifically, the Visual State Space (VSS) block is introduced as the foundation block to capture extensive contextual information, and an asymmetrical encoder-decoder structure is constructed. We conduct comprehensive experiments on the ISIC17, ISIC18, and Synapse datasets, and the results indicate that VM-UNet performs competitively in medical image segmentation tasks. To our best knowledge, this is the first medical image segmentation model constructed based on the pure SSM-based model. We aim to establish a baseline and provide valuable insights for the future development of more efficient and effective SSM-based segmentation systems.

## 0. Main Environments
```bash
conda create -n vmunet python=3.8
conda activate vmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```
The .whl files of causal_conv1d and mamba_ssm could be found here. {[Baidu](https://pan.baidu.com/s/1Tibn8Xh4FMwj0ths8Ufazw?pwd=uu5k) or [GoogleDrive](https://drive.google.com/drive/folders/1ZJjc7sdyd-6KfI7c8R6rDN8bcTz3QkCx?usp=sharing)}

## 1. Prepare the dataset

### ISIC datasets
- The ISIC17 and ISIC18 datasets, divided into a 7:3 ratio, can be found here {[Baidu](https://pan.baidu.com/s/1Y0YupaH21yDN5uldl7IcZA?pwd=dybm)}.

- After downloading the datasets, you are supposed to put them into './data/isic17/' and './data/isic18/', and the file format reference is as follows. (take the ISIC17 dataset as an example.)

- './data/isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png

### Synapse datasets

- For the Synapse dataset, you could follow [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) to download the dataset, or you could download them from {[Baidu](https://pan.baidu.com/s/1JCXBfRL9y1cjfJUKtbEhiQ?pwd=9jti)}.

- After downloading the datasets, you are supposed to put them into './data/Synapse/', and the file format reference is as follows.

- './data/Synapse/'
  - lists
    - list_Synapse
      - all.lst
      - test_vol.txt
      - train.txt
  - test_vol_h5
    - casexxxx.npy.h5
  - train_npz
    - casexxxx_slicexxx.npz

## 2. Prepare the pre_trained weights

- The weights of the pre-trained VMamba could be downloaded from [Baidu](https://pan.baidu.com/s/1ci_YvPPEiUT2bIIK5x8Igw?pwd=wnyy) or [GoogleDrive](https://drive.google.com/drive/folders/1ZJjc7sdyd-6KfI7c8R6rDN8bcTz3QkCx?usp=sharing). After that, the pre-trained weights should be stored in './pretrained_weights/'.



## 3. Train the VM-UNet
```bash
cd VM-UNet
python train.py  # Train and test VM-UNet on the ISIC17 or ISIC18 dataset.
python train_synapse.py  # Train and test VM-UNet on the Synapse dataset.
```

## 3.1 Research tools and completed experiments

`train_full.py` records per-epoch metrics, selects checkpoints by validation pooled Dice, and supports explicit `--resume`. `analyze_full.py` exports per-image Dice/IoU, boundary F1, HD95 in resized-grid pixels, size groups, and failure panels. Validation scores are development results, not independent test results.

The completed fully supervised baseline scored **89.53% ± 0.57%** across three seeds; fixed boundary weighting scored **89.50% ± 0.62%**. Context/teacher continuation experiments are complete, and MambaLiteUNet has been abandoned. Their results are summarized in the [research record](docs/后续研究路线.md); they are not queued for further training. Historical model/checkpoint support remains in the code.

The earlier scan-aware semi-supervised scripts (`train_ssl.py`, `eval_cross_domain.py`) remain available as historical experiments. Their direction is paused; the [archived plan](docs/archive/扫描半监督路线_已暂停.md) is not an execution queue.

**NOTE**: If you want to use the trained checkpoint for inference testing only and save the corresponding test images, you can follow these steps:

- **In `config_setting`**:
   - Set the parameter `only_test_and_save_figs` to `True`.
   - Fill in the path of the trained checkpoint in `best_ckpt_path`.
   - Specify the save path for test images in `img_save_path`.

- **Execute the script**:
   After setting the above parameters, you can run `train.py`.

## 4. Obtain the outputs
- After trianing, you could obtain the results in './results/'

## 5. Trained VM-UNet Checkpoint

- You can also obtain our trained VM-UNet on ISIC17, ISIC18 and Synapse from [Baidu Netdisk](https://pan.baidu.com/s/1lygUOFo6fMF_wS_dskwpBQ?pwd=5z00) or [GoogleDrive](https://drive.google.com/drive/folders/1ZJjc7sdyd-6KfI7c8R6rDN8bcTz3QkCx?usp=sharing).

## 6. Acknowledgments

- We thank the authors of [VMamba](https://github.com/MzeroMiko/VMamba) and [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) for their open-source codes.

