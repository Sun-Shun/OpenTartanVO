# OpenTrain — TartanVO Open Reproduction

> 经典深度学习视觉里程计框架 TartanVO 的开源复现与工程优化版本，支持分阶段端到端单目视觉里程计训练与测试。

**主要贡献者：** Shunwang Sun · Jialu Zhang · Tingxi Xue

---

## 📖 项目简介

### 关于 TartanVO

[TartanVO](https://github.com/castacks/tartanvo) 是一种具有强跨域泛化能力的学习型视觉里程计模型，核心贡献包括：

- **跨域泛化：** 仅使用大规模合成数据集（TartanAir）训练，无需任何微调，即可直接泛化至 KITTI、EuRoC 等真实世界数据集。
- **技术创新：** 提出了尺度不变损失函数（Up-to-Scale Loss），并将相机内参作为特征输入融入网络。
- **鲁棒性：** 在各类挑战性轨迹与极端场景下，性能显著优于传统几何方法。

### 关于 OpenTrain

OpenTrain 对 TartanVO 进行了模块化重构，填补了原始框架在训练代码上的空白，提供：

- 清晰的数据加载流水线
- 解耦的光流 / 位姿训练逻辑
- 完整的评测脚本

旨在为深度学习 VO 领域研究者提供规范、易用的 Baseline 工程。

---

## ⚠️ 重要注意事项

在运行或调试代码前，请检查 `utils.py` 第 517 行的 `make_intrinsics_layer` 函数：

- 须确保 `hh` 与 `ww` 的维度映射正确：
  - `ww` → `0`（图像宽度）
  - `hh` → `0`（图像高度）
- 当前代码已在 **PyTorch 1.12** 及其衍生版本下测试通过，其他版本请自行验证。

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

在运行前，请将代码中的数据集根目录修改为本机绝对路径。

### 模型训练

**① 仅训练光流网络**

```bash
python train.py \
    --is_train True --only_flow True --only_pose False --vo False \
    --datastr tartanair \
    --batch_size 1 --num_workers 1 \
    --logs_dir ./runs_test \
    --flow_model ./models/flow/raft-small.pth
```

**② 仅训练位姿网络**

```bash
python train.py \
    --is_train True --only_flow False --only_pose True --vo False \
    --datastr tartanair \
    --batch_size 1 \
    --pose_model ./models/only_pose/single_pose_model.train
```

**③ 完整端到端 VO 训练**

```bash
python train.py \
    --is_train True --only_flow False --only_pose False --vo True \
    --datastr tartanair \
    --batch_size 1
```

### 模型测试与评估

在 `test.py` 的 `__main__` 函数中配置以下参数：

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `data_type` | `tartanair` / `euroc` / `kitti` | 评估数据集类型 |
| `model_path` | 绝对路径字符串 | 预训练权重路径 |
| `type` | `flow` / `pose` / `vo` | 测试的网络类型 |

配置完成后执行：

```bash
python test.py
```

---

## 📊 评估指标与结果

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
