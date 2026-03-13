# TartanVO-Open

An open-source implementation of **TartanVO** — a generalizable learning-based monocular visual odometry system trained on the [TartanAir](https://theairlab.org/tartanair-dataset/) dataset and evaluated on TartanAir, KITTI, and EuRoC benchmarks.

> Paper: [TartanVO: A Generalizable Learning-based VO](https://arxiv.org/abs/2011.00359)  
> Original code: https://github.com/castacks/tartanvo

---

## Requirements

- Python ≥ 3.6
- PyTorch ≥ 1.3
- CUDA-capable GPU

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Download the Pre-trained Model

Download `tartanvo_1914.pkl` from the [TartanAir model zoo](https://cmu.box.com/s/znk3nq5y0nnc4jvhq2gv8hcz7lqvjxgq) and place it in the `models/` directory.

---

## Running TartanVO

```bash
python vo_trajectory_from_folder.py \
    --model-name tartanvo_1914.pkl \
    --test-dir /path/to/image/folder \
    --pose-file /path/to/gt_pose.txt \   # optional, used for scale & evaluation
    --batch-size 1
```

Additional flags:

| Flag | Description |
|------|-------------|
| `--kitti` | Use KITTI camera intrinsics |
| `--euroc` | Use EuRoC camera intrinsics |
| `--kitti-intrinsics-file <calib.txt>` | Load per-sequence KITTI intrinsics |
| `--save-flow` | Save optical flow images to `results/` |
| `--image-width` / `--image-height` | Crop size (default 640 × 448) |

---

## Project Structure

```
TartanVO-Open/
├── TartanVO.py                     # Main VO inference class
├── vo_trajectory_from_folder.py    # Command-line entry point
├── requirements.txt
├── models/                         # Place pre-trained .pkl files here
├── results/                        # Output trajectories and figures
├── Network/
│   ├── PWC.py                      # PWC-DC optical-flow network
│   ├── VOFlowNet.py                # Pose regression network
│   ├── VONet.py                    # Combined VO network
│   └── correlation.py              # Cost-volume correlation (pure PyTorch)
├── Datasets/
│   ├── tartanTrajFlowDataset.py    # Dataset loader
│   ├── transformation.py           # SE(3) / quaternion utilities
│   └── utils.py                    # Image transforms, visualisation helpers
└── evaluator/
    ├── tartanair_evaluator.py      # High-level evaluator
    ├── evaluator_base.py
    ├── transformation.py
    ├── trajectory_transform.py
    ├── evaluate_ate_scale.py       # ATE metric
    ├── evaluate_rpe.py             # RPE metric
    └── evaluate_kitti.py           # KITTI odometry metric
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| ATE    | Absolute Trajectory Error (with optional scale alignment for monocular) |
| RPE    | Relative Pose Error |
| KITTI  | KITTI odometry benchmark rotation & translation errors |

---

## License

BSD License — see individual source files for details.