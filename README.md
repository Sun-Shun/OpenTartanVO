# OpenTrain (TartanVO Reproduction)

欢迎来到 **OpenTrain**！本项目是对 TartanVO 视觉里程计框架的开源复现与优化。我们支持分阶段的端到端单目视觉里程计训练与测试（包括光流网络、位姿网络以及完整的 VO 联合调试）。

**主要贡献者:** Shunwang Sun, Jialu Zhang, Tingxi Xue

---

## ⚠️ 注意事项 (Important Note)

在运行或调试代码前，请务必检查 `utils.py` 文件中第 517 行的 `make_intrinsics_layer` 函数：
* 须确保 `hh` 和 `ww` 的对应关系正确。
* 目前代码已在 **PyTorch 1.12** 及其衍生版本下测试通过。如果你使用其他版本的 PyTorch，请务必调试确认。
* **正确的维度映射应为**：`ww` 对应 `0`（图像宽度），`hh` 对应 `0`（图像高度）。

---

## 🛠️ 环境依赖 (Environment Requirements)

推荐使用 Conda 来管理运行环境。你可以通过以下命令快速配置与我们一致的 `tartanvopen` 虚拟环境：

```bash
# 1. 创建并激活 conda 环境
conda create -n tartanvopen python=3.10
conda activate tartanvopen

# 2. 安装 PyTorch 及其核心组件 (基于 CUDA 11.8)
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --extra-index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 3. 安装其他核心数据处理与视觉库
pip install numpy==2.2.6 opencv-python==4.13.0.92 scipy==1.15.3 matplotlib==3.10.8 tensorboard==2.20.0

📂 数据集目录结构 (Dataset Structure)
本项目以 KITTI / TartanAir 数据集结构为例，请确保你的数据存放在以下严格的目录结构中：

Plaintext
Dataset_Root
│  ├─train
│  │  ├──train_flow
│  │  │  └──01_0000flow
│  │  │      └──flow.npy
│  │  ├──train_img
│  │  │  └──......
│  │  ├──train_pose
│  │  │  └──......
│  │  └──train_mask (optional)
│  ├─test
│  │  ├──test_flow
│  │  ├──test_img
│  │  ├──test_pose
│  │  └──test_mask (optional)
🚀 运行代码 (Execution)
本项目支持分阶段训练（仅光流网络、仅位姿网络）以及完整的端到端 VO 联合训练。在运行前，请确保修改代码中的数据集根目录绝对路径。

1. 模型训练 (Training)
训练光流网络 (Train Flow Network only):

Bash
python train.py --is_train True --only_flow True --only_pose False --vo False \
    --datastr tartanair \
    --batch_size 1 --num_workers 1 \
    --logs_dir ./runs_test \
    --flow_model ./models/flow/raft-small.pth
训练位姿网络 (Train Pose Network only):

Bash
python train.py --is_train True --only_flow False --only_pose True --vo False \
    --datastr tartanair \
    --batch_size 1 \
    --pose_model ./models/only_pose/single_pose_model.train
完整的端到端 VO 训练 (Train full VO Network):

Bash
python train.py --is_train True --only_flow False --only_pose False --vo True \
    --datastr tartanair \
    --batch_size 1
2. 模型测试与评估 (Testing)
目前的测试配置在 test.py 文件的 __main__ 函数中。在运行测试前，请打开 test.py 并根据需要修改以下参数：

data_type: 评估的数据集类型（可选 'tartanair', 'euroc', 'kitti'）。

model_path: 预训练模型权重的绝对路径。

type: 测试的网络类型（可选 "flow", "pose", "vo"）。

修改完成后，执行测试：

Bash
python test.py
📊 评估指标与结果展示 (Evaluation & Results)
光流评估 (Flow): 程序将输出端点误差 (Endpoint Error, EPE)。

轨迹评估 (Trajectory): 程序会自动计算绝对轨迹误差 (Absolute Trajectory Error, ATE)。

可视化输出: 测试完成后，绝对位姿对齐的轨迹对比图将自动保存在项目根目录的 results/ 文件夹下（例如：results/test_01_0000flow.png），并在图表标题中展示当前的 ATE 分数。

(注意：在运行测试代码前，请确保项目根目录下已手动创建 results/ 文件夹，否则可能会在保存轨迹图时引发错误。)
