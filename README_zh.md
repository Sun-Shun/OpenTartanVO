<h1 align="center">OpenTartanVO — TartanVO Open Reproduction</h1>

<p align="center">Learning-based 单目视觉里程计框架 TartanVO 的开源复现与工程优化版本，支持分阶段端到端单目视觉里程计训练与测试。</p>

<p align="center"><strong>Contributors:</strong> <a href="https://github.com/zhangcv123/">Jialu Zhang</a> · <a href="https://sun-shun.github.io/">Shunwang Sun</a> · Tingxi Xue&emsp;&emsp;&emsp;&emsp;<a href="README.md">English</a></p>


---

## 🎉 最新论文发布

> **Analogy-Augmented Uncertainty-aware Monocular Visual Odometry System**
>
> 📄 [IEEE Xplore →](https://ieeexplore.ieee.org/abstract/document/11396037)

本项目是上述论文的基础复现框架。在此之上，我们提出了 **CUVO** 网络与**类比增强（Analogy Augmentation）**策略：

- **CUVO 网络：** 利用语义感知注意力机制，主动抑制高不确定性区域的干扰，提升位姿估计精度。
- **类比增强与一致性：** 引入时序翻转、随机旋转和几何镜像，并配合一致性损失稳定训练。在仅 27k 数据下，TartanAir 和 KITTI 的零样本泛化分别最高提升 29.5% 和 23.3%。

---

## 📖 项目简介

本项目（**OpenTartanVO**）对具备强跨域泛化能力的 **TartanVO** 进行了模块化重构与工程优化，旨在为研究者提供规范、易用的 Baseline 工程：

- **跨域泛化与鲁棒性：** 仅依赖合成数据集（TartanAir）训练即可直接泛化至真实世界（如 KITTI），在挑战性轨迹下远超传统几何方法。
- **完备的工程框架：** 填补了原版训练代码空白，提供清晰的数据加载流水线、解耦的光流/位姿训练逻辑及完整的评测脚本。

### 光流网络说明

本仓库开源的光流骨干网络为 **RAFT**（Recurrent All-Pairs Field Transforms）：

> Teed & Deng, *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow*, ECCV 2020
> [📄 arXiv](https://arxiv.org/abs/2003.12039) · [💻 GitHub](https://github.com/princeton-vl/RAFT)

下方评估结果（Table XI / XII）中出现的 **TartanVO (Sea-RAFT)** 系列，来自我们后续论文 *Analogy-Augmented Uncertainty-aware Monocular Visual Odometry* 的实验，该论文将光流网络替换为 [**Sea-RAFT**](https://github.com/princeton-vl/SEA-RAFT) 以探究更强骨干网络对性能的影响，**不属于本仓库开源代码的范围**：

> Wang et al., *Sea-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow*, ECCV 2024
> [📄 arXiv](https://arxiv.org/abs/2405.14793) · [💻 GitHub](https://github.com/princeton-vl/SEA-RAFT)

---

## 🔧 核心训练流程

OpenTartanVO 完整复现了端到端训练的四大核心模块：

1. **两阶段网络架构：**
   - **匹配网络 $M_\theta$（光流网络）：** 估计连续两帧的稠密光流，本仓库使用 [**RAFT**](https://github.com/princeton-vl/RAFT)，输出尺寸为 $H/4 \times W/4$。
   - **位姿网络 $P_\phi$：** 以光流与相机内参层为输入，回归相对位姿 $\delta_t^{t+1} = (T, R)$。
2. **分阶段训练策略：** - **阶段一：** 使用真值光流单独优化位姿网络 $P_\phi$。
   - **阶段二：** 联合光流与位姿网络，进行端到端优化，总损失为：
   $$\mathcal{L} = \lambda \mathcal{L}_f + \mathcal{L}_p$$
3. **尺度不变损失（Up-to-Scale Loss）：** 单目 VO 存在固有尺度歧义，网络仅预测平移方向，通过归一化距离损失消除尺度问题：
   $$\mathcal{L}_p^{norm} = \left\| \frac{\hat{T}}{\max(\|\hat{T}\|, \epsilon)} - \frac{T}{\max(\|T\|, \epsilon)} \right\| + \|\hat{R} - R\|$$
4. **相机内参层（Intrinsics Layer）：** 将相机内参编码为二通道特征图并与光流拼接，结合随机裁剪与缩放（RCR）策略，使模型免微调即可适配多种不同焦距的相机。

---

## 🛠️ 环境配置

推荐使用 Conda 管理运行环境。

```bash
# 1. 创建并激活虚拟环境
conda create -n tartanvopen python=3.10
conda activate tartanvopen

# 2. 安装 PyTorch（基于 CUDA 11.8）
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# 3. 安装核心依赖库
pip install numpy==2.2.6 opencv-python==4.13.0.92 scipy==1.15.3 \
    matplotlib==3.10.8 tensorboard==2.20.0
```

---

## 📂 数据集目录结构

以 KITTI / TartanAir 为例，请严格按照以下结构存放数据：

```
Dataset_Root/
├── train/
│   ├── train_flow/
│   │   └── 01_0000flow/
│   │       └── flow.npy
│   ├── train_img/
│   ├── train_pose/
│   └── train_mask/          # 可选
└── test/
    ├── test_flow/
    ├── test_img/
    ├── test_pose/
    └── test_mask/            # 可选
```

---

## 🚀 运行指南

### 参数说明

**train.py 参数**

| 参数 | 必填 | 默认值 | 说明 |
|------|:----:|--------|------|
| `--data_root` | ✅ | — | 数据集根目录（须包含 `train/` 和 `test/`）|
| `--only_flow` | — | `False` | 仅训练光流网络 |
| `--only_pose` | — | `False` | 仅训练位姿网络 |
| `--vo` | — | `False` | 完整端到端 VO 训练 |
| `--flow_model` | — | `None` | 光流预训练权重路径 |
| `--pose_model` | — | `None` | 位姿预训练权重路径 |
| `--datastr` | — | `tartanair` | 数据集类型：`tartanair` / `euroc` / `kitti` |
| `--logs_dir` | — | `./runs_test` | TensorBoard 日志目录 |
| `--root_path` | — | `./models` | 模型保存目录 |
| `--batch_size` | — | `1` | 批大小 |
| `--num_workers` | — | `1` | DataLoader 工作进程数 |
| `--sample_step` | — | `1` | 数据抽样步长（调试时可设为 200）|

> `--only_flow`、`--only_pose`、`--vo` 三者必须恰好有一个为 `True`。

**test.py 参数**

| 参数 | 必填 | 默认值 | 说明 |
|------|:----:|--------|------|
| `--data_root` | ✅ | — | 数据集根目录（须包含 `test/`）|
| `--model_path` | ✅ | — | 预训练模型权重路径 |
| `--datastr` | — | `tartanair` | 数据集类型：`tartanair` / `euroc` / `kitti` |
| `--test_mode` | — | `vo` | 测试模式：`flow` / `pose` / `vo` |
| `--results_dir` | — | `./results` | 轨迹图保存目录（自动创建）|

---

### 模型训练

**① 仅训练光流网络**

```bash
python train.py \
    --data_root /path/to/dataset \
    --only_flow True \
    --flow_model ./models/flow/raft-small.pth \
    --datastr tartanair \
    --batch_size 64 --num_workers 1 \
    --logs_dir ./runs_test
```

**② 仅训练位姿网络**

```bash
python train.py \
    --data_root /path/to/dataset \
    --only_pose True \
    --pose_model ./models/only_pose/single_pose_model.train \
    --datastr tartanair \
    --batch_size 128
```

**③ 完整端到端 VO 训练**

```bash
python train.py \
    --data_root /path/to/dataset \
    --vo True \
    --datastr tartanair \
    --batch_size 128
```

### 模型测试与评估

```bash
python test.py \
    --data_root /path/to/dataset \
    --model_path /path/to/model.pth \
    --datastr tartanair \
    --test_mode vo \
    --results_dir ./results
```

---

## 📊 评估指标与结果

### 可复现性分析（Reproducibility Analysis）

我们以 [Sea-RAFT](https://github.com/princeton-vl/SEA-RAFT) 替换 RAFT 作为光流骨干网络（该替换实验属于后续论文 *Analogy-Augmented Uncertainty-aware Monocular Visual Odometry*，不在本仓库开源代码范围内），在严格遵循原始训练流程与损失函数（包含光流网络微调）的前提下进行复现。结果表明，使用 Sea-RAFT 的复现版本在 KITTI 与 TartanAir 基准上均达到或超越原始 PWC-Net 实现水平。

> **模型权重：** 本仓库不提供预训练模型权重。如有需要，请发送邮件至 [shunwang_sun@163.com](mailto:shunwang_sun@163.com) 获取。

> **关于冻结光流骨干网络的说明：** 我们在 CUVO 中选择冻结 Sea-RAFT 骨干网络权重，原因在于联合微调重型光流网络与位姿估计器对 GPU 显存与训练时间消耗极大，结合 Analogy Augmentation 策略后在当前硬件上已不可行。验证实验表明，冻结模型在大多数序列上优于微调版本，平均 ATE 差异极小，仅在少数困难序列（如 KITTI 01、06 及 TartanAir ME002、MH000）上略有下降。

---

#### Table XI — KITTI 数据集（ATE，越低越好）

> <u>**加粗下划线**</u> = 最优 &nbsp;|&nbsp; **加粗** = 次优 &nbsp;|&nbsp; `*` = 光流网络权重已冻结（无微调）

| Methods | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TartanVO (PWC-Net) | 69.11 | 53.19 | 78.78 | 2.70 | **1.99** | 55.18 | <u>**10.50**</u> | 13.87 | 48.16 | 27.93 | 11.90 | <u>**33.94**</u> |
| TartanVO (Sea-RAFT) | **65.08** | **53.03** | 76.27 | **2.69** | **1.99** | 52.21 | **10.47** | 13.71 | 47.37 | 27.73 | 11.88 | **32.95** |
| TartanVO (Sea-RAFT) \* **Our Baseline** | 76.91 | 144.58 | **56.44** | 3.61 | <u>**3.48**</u> | **22.61** | 57.92 | **7.32** | **39.97** | **24.74** | **9.78** | 40.67 |

---

#### Table XII — TartanAir 数据集（ATE，越低越好）

| Methods | ME000 | ME001 | ME002 | ME003 | ME004 | ME005 | ME006 | ME007 | MH000 | MH001 | MH002 | MH003 | MH004 | MH005 | MH006 | MH007 | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TartanVO (PWC-Net) | 27.30 | <u>**0.86**</u> | **0.64** | 7.18 | <u>**2.02**</u> | **0.58** | 4.12 | **0.42** | 2.12 | **0.31** | 1.28 | 1.09 | 0.99 | **1.40** | 1.74 | **1.42** | 3.34 |
| TartanVO (Sea-RAFT) | <u>**12.76**</u> | <u>**0.86**</u> | **0.64** | **7.07** | **2.01** | **0.58** | **4.06** | **0.42** | <u>**2.10**</u> | **0.31** | **1.27** | <u>**1.08**</u> | **0.93** | <u>**1.39**</u> | **1.72** | <u>**1.41**</u> | **2.41** |
| TartanVO (Sea-RAFT) \* **Our Baseline** | **11.34** | **0.45** | <u>**1.82**</u> | 11.61 | 3.71 | <u>**0.62**</u> | <u>**3.20**</u> | <u>**0.54**</u> | 3.05 | <u>**0.33**</u> | <u>**1.04**</u> | **0.64** | <u>**0.80**</u> | 1.83 | <u>**1.56**</u> | 1.50 | **2.75** |

---

### 输出文件说明

| 任务 | 指标 | 输出说明 |
|------|------|----------|
| 光流评估 | EPE（端点误差） | 终端直接输出 |
| 轨迹评估 | ATE（绝对轨迹误差） | 自动计算并展示 |
| 可视化 | 轨迹对比图（PNG） | 保存至 `results/` 目录，标题含 ATE 分数 |

> **注意：** 运行测试前，请手动在项目根目录创建 `results/` 文件夹，否则保存轨迹图时将报错。
>
> ```bash
> mkdir -p results
> ```

---

## 📄 许可证 (License)

本项目采用 **BSD 3-Clause 许可证** 进行开源，详情请参见 [LICENSE](LICENSE) 文件。

Copyright (c) 2026, Zhejiang University

---

## 📄 引用 (Citation)

如果本项目对你的研究有所帮助，请引用以下论文：

**CUVO（本仓库对应论文）**

```bibtex
@article{li2026analogy,
  title={Analogy-Augmented Uncertainty-aware Monocular Visual Odometry},
  author={Li, Jituo and Sun, Shunwang and Xue, Tingxi and Liu, Xinqi and Zhang, Jialu and Dong, Huixu and Lu, Guodong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026},
  publisher={IEEE}
}
```

**OpenTartanVO（本仓库复现框架）**

```bibtex
@misc{opentrain2025,
  title     = {OpenTartanVO: An Open-Source Reproduction and Engineering Optimization of TartanVO},
  author    = {Zhang, Jialu and Sun, Shunwang and Xue, Tingxi},
  year      = {2025},
  howpublished = {\url{https://github.com/Sun-Shun/OpenTartanVO}}
}
```

**TartanVO**

```bibtex
@article{tartanvo2020corl,
  title     = {TartanVO: A Generalizable Learning-based VO},
  author    = {Wang, Wenshan and Hu, Yaoyu and Scherer, Sebastian},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2020}
}
```

**TartanAir Dataset**

```bibtex
@article{tartanair2020iros,
  title     = {TartanAir: A Dataset to Push the Limits of Visual SLAM},
  author    = {Wang, Wenshan and Zhu, Delong and Wang, Xiangwei and Hu, Yaoyu and Qiu, Yuheng and Wang, Chen and Hu, Yafei and Kapoor, Ashish and Scherer, Sebastian},
  booktitle = {2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2020}
}
```

More technical details are available in the [TartanVO paper](https://arxiv.org/abs/2011.00359).
