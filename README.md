# OpenTartanVO — TartanVO Open Reproduction

> 经典深度学习视觉里程计框架 TartanVO 的开源复现与工程优化版本，支持分阶段端到端单目视觉里程计训练与测试。

**主要贡献者：** Shunwang Sun · Jialu Zhang · Tingxi Xue

---

## 🎉 最新论文发布

> **Analogy-Augmented Uncertainty-aware Monocular Visual Odometry System**
>
> 📄 [IEEE Xplore →](https://ieeexplore.ieee.org/abstract/document/11396037)

本仓库是上述论文的基础复现框架。在 OpenTartanVO 之上，我们进一步提出了 **CUVO**（Context Attention Uncertainty-aware VO Network）及一套**类比增强（Analogy Augmentation）**数据策略：

- **CUVO 网络：** 引入语义感知注意力机制，主动抑制高不确定性区域的干扰，提升位姿估计精度。
- **类比增强：** 时序翻转、随机旋转、几何镜像三类增强方式同步变换图像对与真值位姿，在仅 27k 数据的受限条件下，零样本泛化能力在 TartanAir 上提升最高 **29.5%**，在 KITTI 上提升最高 **23.3%**。
- **类比一致性损失：** 约束原始数据与增强数据的输出一致性，进一步稳定训练。

CUVO 在 TartanAir 与 KITTI 基准上均超越了此前端到端 VO 方法，并首次提出了面向视觉里程计的图像-位姿数据增强范式。

---

## 📖 项目简介

### 关于 TartanVO

[TartanVO](https://github.com/castacks/tartanvo) 是一种具有强跨域泛化能力的学习型视觉里程计模型，核心贡献包括：

- **跨域泛化：** 仅使用大规模合成数据集（TartanAir）训练，无需任何微调，即可直接泛化至 KITTI、EuRoC 等真实世界数据集。
- **技术创新：** 提出了尺度不变损失函数（Up-to-Scale Loss），并将相机内参作为特征输入融入网络。
- **鲁棒性：** 在各类挑战性轨迹与极端场景下，性能显著优于传统几何方法。

### 关于 OpenTartanVO

OpenTartanVO 对 TartanVO 进行了模块化重构，填补了原始框架在训练代码上的空白，提供：

- 清晰的数据加载流水线
- 解耦的光流 / 位姿训练逻辑
- 完整的评测脚本

旨在为深度学习 VO 领域研究者提供规范、易用的 Baseline 工程。

---

## 🔧 核心训练流程（来自 TartanVO 论文）

OpenTartanVO 完整复现了 TartanVO 的端到端训练流程，涵盖以下四个核心技术模块：

### 1. 两阶段网络架构

TartanVO 由两个串联子模块组成：

- **匹配网络 $M_\theta$（光流网络）：** 以两帧连续 RGB 图像 $I_t, I_{t+1}$ 为输入，估计稠密光流 $F_t^{t+1}$。原版使用 PWC-Net，本仓库支持替换为 Sea-RAFT 以获得更好性能。输出尺寸为 $H/4 \times W/4$。
- **位姿网络 $P_\phi$（Pose Network）：** 以光流与相机内参层为输入，回归相对位姿 $\delta_t^{t+1} = (T, R)$，其中 $T \in \mathbb{R}^3$ 为平移，$R \in so(3)$ 为旋转。骨干网络为去除 BatchNorm 的 ResNet50，分设平移头与旋转头。

### 2. 分阶段训练策略

训练分为两个阶段，本仓库均予以支持：

**阶段一 — 单独训练位姿网络**（使用真值光流）
> 以真值光流 $F_t^{t+1}$ 作为输入，单独优化 $P_\phi$，使位姿回归先收敛到合理解。

**阶段二 — 联合端到端训练**（光流网络 + 位姿网络）
> 将 $M_\theta$ 与 $P_\phi$ 串联，以图像对为输入进行端到端联合优化，端到端总损失为：
>
> $$\mathcal{L} = \lambda \mathcal{L}_f + \mathcal{L}_p$$
>
> 其中 $\mathcal{L}_f$ 为光流估计损失，$\mathcal{L}_p$ 为位姿损失，$\lambda$ 为平衡超参数。

### 3. 尺度不变损失函数（Up-to-Scale Loss）

单目 VO 存在固有的尺度不确定性，直接回归带尺度的平移会导致跨数据集泛化失败。TartanVO 仅预测平移方向，通过归一化距离损失消除尺度歧义：

$$\mathcal{L}_p^{norm} = \left\| \frac{\hat{T}}{\max(\|\hat{T}\|, \epsilon)} - \frac{T}{\max(\|T\|, \epsilon)} \right\| + \|\hat{R} - R\|$$

测试时，平移向量的尺度由真值对齐恢复。本仓库完整实现该损失函数。

### 4. 相机内参层（Intrinsics Layer，IL）

为实现跨相机泛化，TartanVO 将相机内参编码为二通道特征图 $K^c \in \mathbb{R}^{2 \times H \times W}$，与光流拼接后送入位姿网络：

$$K^c_x = (X_{ind} - o_x) / f_x, \quad K^c_y = (Y_{ind} - o_y) / f_y$$

其中 $X_{ind}, Y_{ind}$ 为像素坐标索引矩阵，$f_x, f_y, o_x, o_y$ 为相机焦距与主点坐标。该设计使模型无需重新训练即可适配不同相机。具体实现见 `utils.py` 中的 `make_intrinsics_layer` 函数。

同时，训练时通过**随机裁剪与缩放（RCR）**模拟多种焦距（覆盖 FoV 40°~90°），并同步更新光流标签与内参层，从而学习对不同相机的鲁棒表示。

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

> 若任务涉及 ROS 通信节点，请确保已正确安装 ROS 2 基础组件并 source 工作空间。

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

所有路径均通过命令行参数传入，无需修改源代码。

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
    --batch_size 1 --num_workers 1 \
    --logs_dir ./runs_test
```

**② 仅训练位姿网络**

```bash
python train.py \
    --data_root /path/to/dataset \
    --only_pose True \
    --pose_model ./models/only_pose/single_pose_model.train \
    --datastr tartanair \
    --batch_size 1
```

**③ 完整端到端 VO 训练**

```bash
python train.py \
    --data_root /path/to/dataset \
    --vo True \
    --datastr tartanair \
    --batch_size 1
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

我们将 TartanVO 与 Sea-RAFT 光流网络集成，在严格遵循原始训练流程与损失函数（包含光流网络微调）的前提下进行复现。结果表明，OpenTartanVO 的复现版本在 KITTI 与 TartanAir 基准上均达到或超越原始实现水平。

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

## 📄 引用 (Citation)

如果本项目对你的研究有所帮助，请引用以下论文：

**CUVO（本仓库对应论文）**

```bibtex
@article{cuvo2024,
  title     = {Analogy-Augmented Uncertainty-aware Monocular Visual Odometry System},
  author    = {Sun, Shunwang and Zhang, Jialu and Xue, Tingxi},
  journal   = {IEEE},
  year      = {2024},
  url       = {https://ieeexplore.ieee.org/abstract/document/11396037}
}
```

**OpenTartanVO（本仓库复现框架）**

```bibtex
@misc{opentrain2024,
  title     = {OpenTartanVO: An Open-Source Reproduction and Engineering Optimization of TartanVO},
  author    = {Sun, Shunwang and Zhang, Jialu and Xue, Tingxi},
  year      = {2024},
  howpublished = {\url{https://github.com/your-repo/opentrain}}
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

---

## 📜 许可证 (License)

本软件基于 **BSD 协议** 开源授权。

Copyright © 2020, Carnegie Mellon University. All rights reserved.

在满足以下条件的前提下，允许以源代码或二进制形式进行再分发与使用（无论是否修改）：

- 源代码的再分发必须保留上述版权声明、本条件列表及以下免责声明。
- 二进制形式的再分发必须在随附的文档或其他材料中复制上述版权声明、本条件列表及以下免责声明。
- 未经事先书面许可，不得使用版权持有人或贡献者的名称为衍生产品背书或推广。

> **免责声明：** 本软件由版权持有人及贡献者"按原样"提供，不附带任何明示或暗示的担保，包括但不限于适销性及特定用途适用性的暗示担保。在任何情况下，版权持有人或贡献者均不对任何直接、间接、偶然、特殊、惩戒性或后果性损害承担责任，即使已被告知此类损害的可能性。
