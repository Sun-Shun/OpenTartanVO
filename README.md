<h1 align="center">OpenTartanVO — TartanVO Open Reproduction</h1>

<p align="center">An open-source reproduction and engineering optimization of the learning-based monocular visual odometry framework TartanVO, supporting staged end-to-end training and evaluation.</p>

<p align="center"><strong>Contributors:</strong> <a href="https://sun-shun.github.io/">Shunwang Sun</a> · Jialu Zhang · Tingxi Xue</p>

<p align="center">
  <a href="README_zh.md">中文文档</a>
</p>

---

## 🎉 Latest Publication

> **Analogy-Augmented Uncertainty-aware Monocular Visual Odometry System**
>
> 📄 [IEEE Xplore →](https://ieeexplore.ieee.org/abstract/document/11396037)

This repository serves as the foundational reproduction framework for the above paper. Building upon it, we propose the **CUVO** network and **Analogy Augmentation** strategies:

- **CUVO Network:** Introduces a semantics-aware attention mechanism to actively suppress high-uncertainty regions, improving pose estimation accuracy.
- **Analogy Augmentation & Consistency:** Applies temporal reversal, random rotation, and geometric mirroring with a consistency loss to stabilize training. With only 27k training samples, zero-shot generalization improves by up to **29.5%** on TartanAir and **23.3%** on KITTI.

---

## 📖 Overview

**OpenTartanVO** provides a modular reconstruction and engineering optimization of **TartanVO**, a learning-based VO model with strong cross-domain generalization. It aims to offer researchers a clean, well-structured baseline:

- **Cross-domain Generalization:** Trained solely on synthetic data (TartanAir), the model generalizes directly to real-world datasets (e.g., KITTI) without any fine-tuning, significantly outperforming geometry-based methods on challenging trajectories.
- **Complete Engineering Framework:** Fills the gap left by the original codebase — providing a clear data loading pipeline, decoupled optical flow / pose training logic, and full evaluation scripts.

### Optical Flow Network

The optical flow backbone open-sourced in this repository is **RAFT** (Recurrent All-Pairs Field Transforms):

> Teed & Deng, *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow*, ECCV 2020
> [📄 arXiv](https://arxiv.org/abs/2003.12039) · [💻 GitHub](https://github.com/princeton-vl/RAFT)

The **TartanVO (Sea-RAFT)** entries in the evaluation tables (Table XI / XII) below come from experiments in our follow-up paper *Analogy-Augmented Uncertainty-aware Monocular Visual Odometry*, which replaces the flow network with [**Sea-RAFT**](https://github.com/princeton-vl/SEA-RAFT) to study the impact of a stronger backbone. **These results are outside the scope of this repository's open-sourced code.**

> Wang et al., *Sea-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow*, ECCV 2024
> [📄 arXiv](https://arxiv.org/abs/2405.14793) · [💻 GitHub](https://github.com/princeton-vl/SEA-RAFT)

---

## 🔧 Core Training Pipeline

OpenTartanVO fully reproduces the four key technical modules of TartanVO's end-to-end training:

1. **Two-stage Network Architecture:**
   - **Matching Network $M_\theta$ (Optical Flow):** Estimates dense optical flow from two consecutive frames. This repository uses [**RAFT**](https://github.com/princeton-vl/RAFT); output resolution is $H/4 \times W/4$.
   - **Pose Network $P_\phi$:** Takes optical flow and the camera intrinsics layer as input to regress the relative pose $\delta_t^{t+1} = (T, R)$.

2. **Staged Training Strategy:**
   - **Stage 1:** Optimize the pose network $P_\phi$ independently using ground-truth optical flow.
   - **Stage 2:** Connect $M_\theta$ and $P_\phi$ end-to-end for joint optimization. The total loss is:
   $$\mathcal{L} = \lambda \mathcal{L}_f + \mathcal{L}_p$$

3. **Up-to-Scale Loss Function:** Monocular VO has inherent scale ambiguity. The network predicts translation direction only, using a normalized distance loss to eliminate the scale problem:
   $$\mathcal{L}_p^{norm} = \left\| \frac{\hat{T}}{\max(\|\hat{T}\|, \epsilon)} - \frac{T}{\max(\|T\|, \epsilon)} \right\| + \|\hat{R} - R\|$$

4. **Camera Intrinsics Layer (IL):** Camera intrinsics are encoded into a two-channel feature map and concatenated with the optical flow. Combined with Random Crop-and-Resize (RCR) augmentation, the model adapts to cameras with varying focal lengths without any fine-tuning.

---

## 🛠️ Environment Setup

We recommend using Conda to manage the environment.

```bash
# 1. Create and activate the virtual environment
conda create -n tartanvopen python=3.10
conda activate tartanvopen

# 2. Install PyTorch (CUDA 11.8)
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# 3. Install core dependencies
pip install numpy==2.2.6 opencv-python==4.13.0.92 scipy==1.15.3 \
    matplotlib==3.10.8 tensorboard==2.20.0
```

---

## 📂 Dataset Directory Structure

Using KITTI / TartanAir as an example, please organize your data strictly as follows:

```
Dataset_Root/
├── train/
│   ├── train_flow/
│   │   └── 01_0000flow/
│   │       └── flow.npy
│   ├── train_img/
│   ├── train_pose/
│   └── train_mask/          # optional
└── test/
    ├── test_flow/
    ├── test_img/
    ├── test_pose/
    └── test_mask/            # optional
```

---

## 🚀 Usage

### Argument Reference

**train.py arguments**

| Argument | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `--data_root` | ✅ | — | Dataset root directory (must contain `train/` and `test/`) |
| `--only_flow` | — | `False` | Train the optical flow network only |
| `--only_pose` | — | `False` | Train the pose network only |
| `--vo` | — | `False` | Full end-to-end VO training |
| `--flow_model` | — | `None` | Path to pre-trained flow model weights |
| `--pose_model` | — | `None` | Path to pre-trained pose model weights |
| `--datastr` | — | `tartanair` | Dataset type: `tartanair` / `euroc` / `kitti` |
| `--logs_dir` | — | `./runs_test` | TensorBoard log directory |
| `--root_path` | — | `./models` | Checkpoint save directory |
| `--batch_size` | — | `1` | Batch size |
| `--num_workers` | — | `1` | DataLoader worker count |
| `--sample_step` | — | `1` | Data sub-sampling step (set to 200 for quick debug runs) |

> Exactly one of `--only_flow`, `--only_pose`, and `--vo` must be `True`.

**test.py arguments**

| Argument | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `--data_root` | ✅ | — | Dataset root directory (must contain `test/`) |
| `--model_path` | ✅ | — | Path to the pre-trained model checkpoint |
| `--datastr` | — | `tartanair` | Dataset type: `tartanair` / `euroc` / `kitti` |
| `--test_mode` | — | `vo` | Evaluation mode: `flow` / `pose` / `vo` |
| `--results_dir` | — | `./results` | Directory for trajectory plots (auto-created) |

---

### Training

**① Flow network only**

```bash
python train.py \
    --data_root /path/to/dataset \
    --only_flow True \
    --flow_model ./models/flow/raft-small.pth \
    --datastr tartanair \
    --batch_size 64 --num_workers 1 \
    --logs_dir ./runs_test
```

**② Pose network only**

```bash
python train.py \
    --data_root /path/to/dataset \
    --only_pose True \
    --pose_model ./models/only_pose/single_pose_model.train \
    --datastr tartanair \
    --batch_size 128
```

**③ Full end-to-end VO**

```bash
python train.py \
    --data_root /path/to/dataset \
    --vo True \
    --datastr tartanair \
    --batch_size 128
```

### Evaluation

```bash
python test.py \
    --data_root /path/to/dataset \
    --model_path /path/to/model.pth \
    --datastr tartanair \
    --test_mode vo \
    --results_dir ./results
```

---

## 📊 Evaluation & Results

### Reproducibility Analysis

In our follow-up paper *Analogy-Augmented Uncertainty-aware Monocular Visual Odometry*, we replace RAFT with [Sea-RAFT](https://github.com/princeton-vl/SEA-RAFT) as the optical flow backbone (this substitution is outside the scope of this repository's open-sourced code), strictly following the original training pipeline and loss functions including flow network fine-tuning. Results show that the Sea-RAFT-based reproduction matches or surpasses the original PWC-Net implementation on both KITTI and TartanAir.

> **Model Weights:** Pre-trained model weights are not included in this repository. If you need them, please contact [shunwang_sun@163.com](mailto:shunwang_sun@163.com).

> **Note on freezing the flow backbone:** In CUVO, we choose to freeze the Sea-RAFT backbone weights, as jointly fine-tuning the heavy flow network and the pose estimator demands excessive GPU memory and training time — infeasible on current hardware when combined with the Analogy Augmentation strategy. Experiments show the frozen model outperforms the fine-tuned version on most sequences, with only marginal average ATE difference; slight degradation is observed on a few difficult sequences (e.g., KITTI 01, 06 and TartanAir ME002, MH000).

---

#### Table XI — KITTI Dataset (ATE ↓, lower is better)

> <u>**Bold + Underline**</u> = Best &nbsp;|&nbsp; **Bold** = Second Best &nbsp;|&nbsp; `*` = Flow network weights frozen (no fine-tuning)

| Methods | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TartanVO (PWC-Net) | 69.11 | 53.19 | 78.78 | 2.70 | **1.99** | 55.18 | <u>**10.50**</u> | 13.87 | 48.16 | 27.93 | 11.90 | <u>**33.94**</u> |
| TartanVO (Sea-RAFT) | **65.08** | **53.03** | 76.27 | **2.69** | **1.99** | 52.21 | **10.47** | 13.71 | 47.37 | 27.73 | 11.88 | **32.95** |
| TartanVO (Sea-RAFT) \* **Our Baseline** | 76.91 | 144.58 | **56.44** | 3.61 | <u>**3.48**</u> | **22.61** | 57.92 | **7.32** | **39.97** | **24.74** | **9.78** | 40.67 |

---

#### Table XII — TartanAir Dataset (ATE ↓, lower is better)

| Methods | ME000 | ME001 | ME002 | ME003 | ME004 | ME005 | ME006 | ME007 | MH000 | MH001 | MH002 | MH003 | MH004 | MH005 | MH006 | MH007 | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TartanVO (PWC-Net) | 27.30 | <u>**0.86**</u> | **0.64** | 7.18 | <u>**2.02**</u> | **0.58** | 4.12 | **0.42** | 2.12 | **0.31** | 1.28 | 1.09 | 0.99 | **1.40** | 1.74 | **1.42** | 3.34 |
| TartanVO (Sea-RAFT) | <u>**12.76**</u> | <u>**0.86**</u> | **0.64** | **7.07** | **2.01** | **0.58** | **4.06** | **0.42** | <u>**2.10**</u> | **0.31** | **1.27** | <u>**1.08**</u> | **0.93** | <u>**1.39**</u> | **1.72** | <u>**1.41**</u> | **2.41** |
| TartanVO (Sea-RAFT) \* **Our Baseline** | **11.34** | **0.45** | <u>**1.82**</u> | 11.61 | 3.71 | <u>**0.62**</u> | <u>**3.20**</u> | <u>**0.54**</u> | 3.05 | <u>**0.33**</u> | <u>**1.04**</u> | **0.64** | <u>**0.80**</u> | 1.83 | <u>**1.56**</u> | 1.50 | **2.75** |

---

### Output Files

| Task | Metric | Output |
|------|--------|--------|
| Flow evaluation | EPE (Endpoint Error) | Printed to terminal |
| Trajectory evaluation | ATE (Absolute Trajectory Error) | Computed and displayed automatically |
| Visualization | Trajectory comparison plot (PNG) | Saved to `results/` with ATE score in title |

> **Note:** Please create the `results/` directory before running evaluation, otherwise saving the trajectory plot will raise an error.
>
> ```bash
> mkdir -p results
> ```

---

## 📄 Citation

If this project is useful for your research, please consider citing:

**CUVO (paper associated with this repository)**

```bibtex
@article{li2026analogy,
  title={Analogy-Augmented Uncertainty-aware Monocular Visual Odometry},
  author={Li, Jituo and Sun, Shunwang and Xue, Tingxi and Liu, Xinqi and Zhang, Jialu and Dong, Huixu and Lu, Guodong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026},
  publisher={IEEE}
}
```

**OpenTartanVO (this reproduction framework)**

```bibtex
@misc{opentrain2025,
  title     = {OpenTartanVO: An Open-Source Reproduction and Engineering Optimization of TartanVO},
  author    = {Sun, Shunwang and Zhang, Jialu and Xue, Tingxi},
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
